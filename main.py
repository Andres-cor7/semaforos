"""Backend: recibe posición + rumbo del vehículo de emergencia, consulta Neon y manda órdenes al Arduino.

Usa las tablas de la base de datos del equipo: traffic_light, evp_data y data_record.
Por defecto la posición NO se guarda: se compara y se descarta (solo se registra en data_record
qué semáforo se activó para qué vehículo, sin coordenadas).

Ejecutar:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("semaforos")

# ---------- Configuración (se lee de .env) ----------
NEON_URL = os.environ["NEON_URL"]
API_TOKEN = os.environ["API_TOKEN"]
SERIAL_PORT = os.getenv("SERIAL_PORT", "")            # vacío = modo simulación (solo imprime)
RADIUS_M = float(os.getenv("RADIUS_M", 150))          # distancia máxima al semáforo
TOL_HEADING = float(os.getenv("TOL_HEADING", 35))     # tolerancia entre rumbo del vehículo y "direction" del semáforo
TOL_AHEAD = float(os.getenv("TOL_AHEAD", 70))         # el semáforo debe estar delante (±grados)
HOLD_SECONDS = float(os.getenv("HOLD_SECONDS", 10))   # tiempo en verde sin nuevas posiciones
STORE_POSITION = os.getenv("STORE_POSITION", "false").lower() == "true"  # guardar lat/lon en evp_data


# ---------- Arduino por puerto serie ----------
class Arduino:
    def __init__(self, port: str):
        self.lock = threading.Lock()
        self.ser = None
        if port:
            import serial  # pyserial
            self.ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # el Uno se reinicia al abrir el puerto
            log.info("Arduino conectado en %s", port)
        else:
            log.info("SERIAL_PORT vacío: modo simulación (no se escribe a ningún Arduino)")

    def send(self, channel: int, cmd: str):
        line = f"{'G' if cmd == 'GREEN' else 'A'}{channel}\n"
        log.info("-> Arduino: %s (%s canal %s)", line.strip(), cmd, channel)
        if self.ser:
            with self.lock:
                self.ser.write(line.encode())


arduino: Arduino | None = None
pool = ConnectionPool(
    NEON_URL,
    min_size=1,
    max_size=5,
    kwargs={"row_factory": dict_row, "prepare_threshold": None},  # compatible con el pooler de Neon
    check=ConnectionPool.check_connection,                        # revisa la conexión (Neon se suspende)
    open=False,
)

# ---------- Estado de cruces con verde activo ----------
state_lock = threading.Lock()
active: dict[int, dict] = {}   # clave de cruce -> {"light": id, "plate": str, "channels": [...], "until": ts}


def release(inter: int):
    """Devuelve el cruce al ciclo normal. Llamar con state_lock tomado."""
    s = active.pop(inter, None)
    if s:
        for ch in s["channels"]:
            arduino.send(ch, "AUTO")


def expirer(stop: threading.Event):
    while not stop.wait(0.5):
        with state_lock:
            for inter, s in list(active.items()):
                if time.time() > s["until"]:
                    release(inter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arduino
    arduino = Arduino(SERIAL_PORT)
    pool.open(wait=True, timeout=30)
    stop = threading.Event()
    threading.Thread(target=expirer, args=(stop,), daemon=True).start()
    yield
    stop.set()
    with state_lock:
        for inter in list(active):
            release(inter)
    pool.close()


app = FastAPI(title="Semáforos EVP", lifespan=lifespan)


# ---------- Geometría (sin PostGIS: la tabla guarda LAT y LON como FLOAT) ----------
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distancia en metros entre dos coordenadas."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2) -> float:
    """Rumbo (0..360, 0 = norte) desde el punto 1 hacia el punto 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def ang_diff(a: float, b: float) -> float:
    """Diferencia angular mínima (0..180)."""
    return abs((a - b + 180) % 360 - 180)


def check_token(authorization: str | None):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Token inválido")


class Position(BaseModel):
    license_plates: str = Field(min_length=1, max_length=10)   # placa del vehículo (evp_data)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    heading: float = Field(ge=0, le=360)   # grados, 0 = norte (mismo criterio que traffic_light.direction)


VEHICLE = "SELECT license_plates, is_operational FROM evp_data WHERE rtrim(license_plates) = %s"

# Prefiltro por cuadro de coordenadas; la distancia exacta se calcula en Python.
LIGHTS = """
SELECT id, name_light, lat, lon, direction, intersection_id, channel
FROM traffic_light
WHERE is_active AND lat BETWEEN %s AND %s AND lon BETWEEN %s AND %s
"""


@app.post("/api/position")
def position(p: Position, authorization: str | None = Header(None)):
    check_token(authorization)
    plate = p.license_plates.strip().upper()
    with pool.connection() as conn:
        veh = conn.execute(VEHICLE, (plate,)).fetchone()
        if not veh or not veh["is_operational"]:
            raise HTTPException(403, "Vehículo no registrado o no operativo")

        if STORE_POSITION:
            conn.execute(
                "UPDATE evp_data SET lat = %s, lon = %s, current_direction = %s WHERE rtrim(license_plates) = %s",
                (p.lat, p.lon, p.heading, plate),
            )

        dlat = RADIUS_M / 111_320
        dlon = RADIUS_M / (111_320 * max(math.cos(math.radians(p.lat)), 0.01))
        rows = conn.execute(LIGHTS, (p.lat - dlat, p.lat + dlat, p.lon - dlon, p.lon + dlon)).fetchall()

        candidates = []
        for r in rows:
            dist = haversine(p.lat, p.lon, r["lat"], r["lon"])
            if dist > RADIUS_M:
                continue
            b = bearing(p.lat, p.lon, r["lat"], r["lon"])
            if ang_diff(r["direction"], p.heading) <= TOL_HEADING and ang_diff(b, p.heading) <= TOL_AHEAD:
                candidates.append((dist, r))
        if not candidates:
            return {"action": "none", "nearby": len(rows)}

        dist, chosen = min(candidates, key=lambda c: c[0])
        inter = chosen["intersection_id"] if chosen["intersection_id"] is not None else -chosen["id"]

        with state_lock:
            cur = active.get(inter)
            if cur and time.time() > cur["until"]:
                release(inter)
                cur = None
            if cur and cur["light"] != chosen["id"]:
                return {"action": "busy", "light_id": chosen["id"]}   # otro vehículo tiene el verde

            if cur is None:   # primera petición: verde para este, los demás al ciclo normal
                if chosen["intersection_id"] is not None:
                    lights = conn.execute(
                        "SELECT id, channel FROM traffic_light WHERE intersection_id = %s AND is_active",
                        (chosen["intersection_id"],),
                    ).fetchall()
                else:
                    lights = [chosen]
                channels = []
                for l in lights:
                    if l["channel"] is None:
                        log.warning("El semáforo %s no tiene 'channel'; no se puede controlar", l["id"])
                        continue
                    channels.append(l["channel"])
                    arduino.send(l["channel"], "GREEN" if l["id"] == chosen["id"] else "AUTO")
                cur = active[inter] = {"light": chosen["id"], "plate": plate, "channels": channels,
                       "until": time.time() + HOLD_SECONDS}
                try:
                    with conn.transaction():   # si falla, solo se descarta el historial
                        conn.execute(
                            "INSERT INTO data_record (traffic_light_id, license_plates_evp) VALUES (%s, %s)",
                            (chosen["id"], veh["license_plates"]),
                        )
                except Exception:
                    log.exception("No se pudo registrar en data_record")
            elif chosen["channel"] is not None:   # latido: el Arduino suelta el verde si dejan de llegar órdenes
                arduino.send(chosen["channel"], "GREEN")
            cur["until"] = time.time() + HOLD_SECONDS

    return {"action": "green", "light_id": chosen["id"], "name": chosen["name_light"],
            "dist_m": round(dist, 1)}


@app.post("/api/test/{channel}/{cmd}")
def test_arduino(channel: int, cmd: str, authorization: str | None = Header(None)):
    """Prueba manual del Arduino sin pasar por GPS. cmd = GREEN | AUTO."""
    check_token(authorization)
    cmd = cmd.upper()
    if cmd not in ("GREEN", "AUTO"):
        raise HTTPException(400, "cmd debe ser GREEN o AUTO")
    arduino.send(channel, cmd)
    return {"ok": True}


@app.get("/api/health")
def health():
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True, "simulacion": SERIAL_PORT == ""}


# La página web del teléfono (static/index.html) se sirve desde el mismo servidor
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")