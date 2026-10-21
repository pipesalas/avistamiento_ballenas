from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.store import DataStore, _day_column, _day_frame, _day_slices
from scripts.prepare_photos import thumbnail


def test_day_lookup_handles_unsorted_dates_repeated_times_and_nat() -> None:
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2024-01-02 00:00", "2024-01-01 12:00", None, "2024-01-01 00:00"]),
        "value": [20, 12, 99, 10],
    })
    prepared = _day_column(frame)
    days = _day_slices(prepared)
    assert set(days) == {"2024-01-01", "2024-01-02"}
    assert _day_frame(prepared, days, "2024-01-01")["value"].tolist() == [10, 12]
    assert _day_frame(prepared, days, "2024-01-02")["value"].tolist() == [20]
    assert _day_frame(prepared, days, "2024-01-03").empty
    empty = _day_column(frame.iloc[:0].copy())
    assert _day_slices(empty) == {}


def test_environment_depths_and_enrichment_preserve_values() -> None:
    store = DataStore.__new__(DataStore)
    store.temperature = _day_column(pd.DataFrame({
        "time": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"]),
        "latitude": [-40.0] * 5,
        "longitude": [-74.0, -74.0, -73.5, -73.5, -74.0],
        "depth": [1.5, 10.0, 1.5, 10.0, 1.5],
        "temperature": [10.0, 14.0, 20.0, np.nan, 99.0],
    }))
    store.chlorophyll = _day_column(pd.DataFrame({
        "time": pd.to_datetime(["2024-01-01"] * 2),
        "latitude": [-40.0, -40.0],
        "longitude": [-74.0, -74.0],
        "chlorophyll": [2.0, 6.0],
        "phyc": [4.0, 8.0],
    }))
    store.min_depth = 1.5
    store._temp_by_day = _day_slices(store.temperature)
    store._chl_by_day = _day_slices(store.chlorophyll)

    def values(variable: str, depth: str = "surface") -> list:
        return [f["properties"]["value"] for f in store.environment(variable, "2024-01-01", depth)["features"]]

    assert values("temperature") == [10.0, 20.0]
    assert values("temperature", "mean") == [12.0, 20.0]
    assert values("temperature", "10") == [14.0]
    assert values("chlorophyll") == [4.0]
    assert values("phyc") == [6.0]
    for variable in ("temperature", "chlorophyll", "phyc"):
        missing = store.environment(variable, "2020-01-01")
        assert missing["features"] == []
        assert missing["properties"]["min"] is None

    sightings = pd.DataFrame({
        "day": ["2024-01-01", "2020-01-01"],
        "latitude": [-40.0, -40.0],
        "longitude": [-74.0, -74.0],
    })
    store._enrich_environment(sightings)
    assert sightings.loc[0, "sst_c"] == 10.0
    assert sightings.loc[0, "chlorophyll"] == 2.0
    assert pd.isna(sightings.loc[1, "sst_c"])
    assert pd.isna(sightings.loc[1, "chlorophyll"])


def test_public_data_compression_and_cache_policy() -> None:
    client = TestClient(app)
    plain = client.get("/api/sightings", headers={"Accept-Encoding": "identity"})
    compressed = client.get("/api/sightings", headers={"Accept-Encoding": "gzip"})
    assert plain.status_code == compressed.status_code == 200
    assert compressed.json() == plain.json()
    assert compressed.headers["Content-Encoding"] == "gzip"
    assert "accept-encoding" in compressed.headers["Vary"].lower()
    assert int(compressed.headers["Content-Length"]) < len(plain.content) / 2
    assert compressed.headers["Cache-Control"] == "public, max-age=300, s-maxage=3600"
    for path in ("/api/sightings?source=invalid", "/api/unknown", "/media/fotos/missing.jpg"):
        error = client.get(path)
        assert error.status_code >= 400
        assert "Cache-Control" not in error.headers

    photo = client.get("/media/logo", headers={"Accept-Encoding": "gzip"})
    assert photo.status_code == 200
    assert "Content-Encoding" not in photo.headers
    assert photo.headers["Cache-Control"] == "public, max-age=86400"


def test_thumbnails_keep_originals_and_support_optional_metadata(tmp_path: Path) -> None:
    boat = tmp_path / "data" / "fotos"
    boat.mkdir(parents=True)
    (tmp_path / "data" / "fotos_conteo_directo").mkdir()
    source = boat / "ballena 01_01_2024.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Camera orientation: rotate 90 degrees clockwise.
    Image.new("RGB", (1000, 500), color="blue").save(source, exif=exif)
    original = source.read_bytes()

    store = DataStore.__new__(DataStore)
    store.root = tmp_path
    store.species = []
    store.by_slug = {}
    store._photos = store._load_photos()
    assert "thumbnail_url" not in store.photos()[0]

    output = boat / "thumbnails"
    output.mkdir()
    destination = output / f"{source.name}.webp"
    thumbnail(source, destination)
    with Image.open(destination) as image:
        assert image.format == "WEBP"
        assert image.size == (320, 640)
    assert source.read_bytes() == original
    store._photos = store._load_photos()
    photos = store.photos()
    assert len(photos) == 1
    assert photos[0]["url"] == "/media/fotos/ballena%2001_01_2024.jpg"
    assert photos[0]["thumbnail_url"] == "/media/fotos/thumbnails/ballena%2001_01_2024.jpg.webp"
