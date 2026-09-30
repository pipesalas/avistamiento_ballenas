from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.store import ROOT, get_store

app = FastAPI(title="Avistamiento de mamíferos marinos")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA = ROOT / "data"
app.mount("/media/fotos", StaticFiles(directory=DATA / "fotos"), name="fotos")
app.mount(
    "/media/fotos_conteo_directo",
    StaticFiles(directory=DATA / "fotos_conteo_directo"),
    name="fotos_orilla",
)


@lru_cache(maxsize=256)
def _environment(variable: str, date: str, depth: str) -> dict:
    try:
        return get_store().environment(variable, date, depth)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.on_event("startup")
def startup() -> None:
    get_store()


@app.get("/media/logo")
def logo() -> FileResponse:
    return FileResponse(DATA / "logo_vuelvealoceano.png")


@app.get("/api/meta")
def meta() -> dict:
    return get_store().meta()


@app.get("/api/environment")
def environment(
    variable: str = Query("temperature"),
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    depth: str = Query("surface"),
) -> dict:
    if variable not in ("temperature", "chlorophyll", "phyc"):
        raise HTTPException(status_code=400, detail="Variable desconocida")
    return _environment(variable, date, depth)


@app.get("/api/sightings")
def sightings(
    source: str = Query("both"),
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    species: str | None = None,
    behavior: str | None = None,
) -> dict:
    if source not in ("boat", "shore", "both"):
        raise HTTPException(status_code=400, detail="Fuente desconocida")
    return get_store().sightings(source, date, date_from, date_to, species, behavior)


@app.get("/api/shore-zones")
def shore_zones(
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    species: str | None = None,
) -> dict:
    return get_store().shore_zones(date, date_from, date_to, species)


@app.get("/api/tracks")
def tracks(date: str | None = None) -> dict:
    return get_store().tracks(date)


@app.get("/api/photos")
def photos(
    source: str = Query("both"),
    date: str | None = None,
    species: str | None = None,
) -> list:
    if source not in ("boat", "shore", "both"):
        raise HTTPException(status_code=400, detail="Fuente desconocida")
    return get_store().photos(source, date, species)


@app.get("/api/stats/species")
def stats_species(source: str = Query("both")) -> list:
    return get_store().stats_species(source)


@app.get("/api/stats/timeline")
def stats_timeline(
    source: str = Query("both"),
    species: str | None = None,
    grain: str = Query("day"),
) -> list:
    if grain not in ("day", "month"):
        raise HTTPException(status_code=400, detail="grain debe ser day o month")
    return get_store().stats_timeline(source, species, grain)


@app.get("/api/stats/seasonality")
def stats_seasonality(source: str = Query("both"), species: str | None = None) -> list:
    return get_store().stats_seasonality(source, species)


@app.get("/api/stats/behavior")
def stats_behavior(species: str | None = None) -> list:
    return get_store().stats_behavior(species)
