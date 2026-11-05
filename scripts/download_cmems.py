"""Download high-resolution satellite SST (MUR 1 km) and Chlorophyll-a (NOAA DINEOF 2 km) for sighting dates.

- Sea Surface Temperature (SST): NASA JPL MUR v4.1 (0.01°, ~1 km)
  Source: https://data.source.coop/ausantarctic/ghrsst-mur-v2 (Cloud-Optimized GeoTIFF mirror of PO.DAAC MUR-JPL-L4-GLOB-v4.1)
- Chlorophyll-a (chlor_a): NOAA CoastWatch Multi-sensor (VIIRS SNPP/NOAA-20 + Sentinel-3A OLCI) DINEOF 2 km Daily
  Source: https://coastwatch.noaa.gov/thredds/ncss/grid/chloraMergedDINEOF2kmGlobalDailyWW00
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import io
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Bounding box covering the Huiro, Chaihuín and Corral study area plus ~35 km buffer.
DEFAULT_BBOX = (-74.15, -40.30, -73.35, -39.60)
MUR_TRANSFORM = Affine(0.01, 0.0, -179.995, 0.0, -0.01, 89.995)
PARQUET_OPTIONS = {"index": False, "compression": "zstd", "compression_level": 9}


def sighting_dates_from_workbook(root: Path = ROOT) -> list[str]:
    matches = sorted((root / "data").glob("Monitoreo*.xlsx"))
    if not matches:
        raise FileNotFoundError("No se encontró el archivo Excel de monitoreo en data/")
    wb = matches[0]

    boat = pd.read_excel(wb, sheet_name="Muestreo a distancia")
    boat.columns = [str(col).strip() for col in boat.columns]
    boat = boat.dropna(subset=["Fecha", "Especie", "Latitud", "Longitud"]).copy()
    boat_days = pd.to_datetime(boat["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d").dropna()
    boat_days = boat_days[boat_days >= "2023-01-01"]

    with pd.ExcelFile(wb) as xl:
        shore_sheet = next(
            (name for name in xl.sheet_names if "conteo directo" in name.lower()),
            xl.sheet_names[-1],
        )
        shore = pd.read_excel(xl, sheet_name=shore_sheet)
    shore.columns = [str(col).strip() for col in shore.columns]
    shore = shore.dropna(subset=["Fecha", "Especie", "Subzona"]).copy()
    shore_days = pd.to_datetime(shore["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d").dropna()

    date_pattern = re.compile(r"(?P<d>\d{2})[_:](?P<m>\d{2})[_:](?P<y>\d{4})")
    media_days: set[str] = set()
    for folder in (root / "data" / "fotos", root / "data" / "fotos_conteo_directo"):
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            if item.is_file() and not item.name.startswith("."):
                m = date_pattern.search(item.name)
                if m:
                    media_days.add(f"{m.group('y')}-{m.group('m')}-{m.group('d')}")

    return sorted(set(boat_days) | set(shore_days) | media_days)


def discover_chl_files(years: list[int]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for year in years:
        url = f"https://coastwatch.noaa.gov/thredds/catalog/chloraMergedDINEOF2kmGlobalDailyWW00/{year}/catalog.xml"
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml = resp.read().decode("utf-8")
        for match in re.finditer(r"(V(\d{4})(\d{3})_[^\"\s]+\.nc)", xml):
            fname, y, doy = match.group(1), int(match.group(2)), int(match.group(3))
            iso = date.fromordinal(date(y, 1, 1).toordinal() + doy - 1).isoformat()
            catalog[iso] = f"{y}/{fname}"
    return catalog


def download_temperature(dates: list[str], bbox: tuple[float, float, float, float], workers: int) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = bbox
    win = from_bounds(lon_min, lat_min, lon_max, lat_max, MUR_TRANSFORM).round_offsets().round_lengths()
    row_off, col_off = int(win.row_off), int(win.col_off)
    height, width = int(win.height), int(win.width)
    lats = (89.995 - (np.arange(row_off, row_off + height) + 0.5) * 0.01).astype("float32")
    lons = (-179.995 + (np.arange(col_off, col_off + width) + 0.5) * 0.01).astype("float32")
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    def fetch_day(day_iso: str) -> pd.DataFrame | None:
        y, m, d = day_iso.split("-")
        url = (
            f"https://data.source.coop/ausantarctic/ghrsst-mur-v2/"
            f"{y}/{m}/{d}/{y}{m}{d}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif"
        )
        for attempt in range(3):
            try:
                with rasterio.open(url) as src:
                    raw = src.read(1, window=win)
                    scale = src.scales[0] if src.scales else 0.001
                    offset = src.offsets[0] if src.offsets else 25.0
                    nodata = src.nodata if src.nodata is not None else -32768.0
                mask = raw != nodata
                if not np.any(mask):
                    return None
                vals = (raw[mask].astype("float32") * np.float32(scale) + np.float32(offset)).round(2)
                return pd.DataFrame(
                    {
                        "time": pd.Timestamp(day_iso),
                        "depth": np.float32(0.0),
                        "latitude": lat_grid[mask],
                        "longitude": lon_grid[mask],
                        "temperature": vals,
                    }
                )
            except Exception as exc:
                if attempt == 2:
                    print(f"[SST] Omitiendo {day_iso}: {exc}", flush=True)
                    return None
                time.sleep(1)
        return None

    frames: list[pd.DataFrame] = []
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
    ):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_day, d): d for d in dates}
            for idx, future in enumerate(as_completed(futures), 1):
                frame = future.result()
                if frame is not None and not frame.empty:
                    frames.append(frame)
                if idx % 100 == 0 or idx == len(dates):
                    print(f"[SST] {idx}/{len(dates)} fechas descargadas", flush=True)

    result = pd.concat(frames, ignore_index=True)
    for col in ("depth", "latitude", "longitude", "temperature"):
        result[col] = result[col].astype("float32")
    return result.sort_values(["time", "depth", "latitude", "longitude"]).reset_index(drop=True)


def download_chlorophyll(dates: list[str], bbox: tuple[float, float, float, float], workers: int) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = bbox
    years = sorted({int(d[:4]) for d in dates})
    catalog = discover_chl_files(years)

    def fetch_day(day_iso: str) -> pd.DataFrame | None:
        rel = catalog.get(day_iso)
        if not rel:
            return None
        url = (
            f"https://coastwatch.noaa.gov/thredds/ncss/grid/chloraMergedDINEOF2kmGlobalDailyWW00/"
            f"{rel}?var=chlor_a&north={lat_max}&west={lon_min}&east={lon_max}&south={lat_min}&accept=netcdf"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    raw = resp.read()
                ds = xr.open_dataset(io.BytesIO(raw), engine="scipy")
                df = ds["chlor_a"].to_dataframe().reset_index()
                ds.close()
                df = df.rename(columns={"lat": "latitude", "lon": "longitude", "chlor_a": "chlorophyll"})
                df = df.dropna(subset=["chlorophyll"])
                df = df[df["chlorophyll"] > 0].copy()
                df["time"] = pd.Timestamp(day_iso)
                df["depth"] = np.float32(0.0)
                return df[["time", "depth", "latitude", "longitude", "chlorophyll"]]
            except Exception as exc:
                if attempt == 2:
                    print(f"[Chl-a] Omitiendo {day_iso}: {exc}", flush=True)
                    return None
                time.sleep(1)
        return None

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_day, d): d for d in dates}
        for idx, future in enumerate(as_completed(futures), 1):
            frame = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
            if idx % 100 == 0 or idx == len(dates):
                print(f"[Chl-a] {idx}/{len(dates)} fechas descargadas", flush=True)

    result = pd.concat(frames, ignore_index=True)
    for col in ("depth", "latitude", "longitude", "chlorophyll"):
        result[col] = result[col].astype("float32")
    return result.sort_values(["time", "depth", "latitude", "longitude"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga SST (NASA MUR 1 km) y Clorofila-a (NOAA DINEOF 2 km) para los días con avistamientos."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=list(DEFAULT_BBOX),
        help="Recorte espacial (por defecto: -74.15 -40.30 -73.35 -39.60).",
    )
    parser.add_argument("--workers", type=int, default=16, help="Descargas simultáneas.")
    args = parser.parse_args()
    bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])

    dates = sighting_dates_from_workbook()
    print(f"Fechas con avistamientos: {len(dates)} ({dates[0]} a {dates[-1]})", flush=True)

    temperature = download_temperature(dates, bbox, args.workers)
    chlorophyll = download_chlorophyll(dates, bbox, args.workers)

    temp_path = DATA / "temperature.parquet"
    chl_path = DATA / "chlorophyll.parquet"
    temperature.to_parquet(temp_path, **PARQUET_OPTIONS)
    chlorophyll.to_parquet(chl_path, **PARQUET_OPTIONS)

    print(
        f"Temperatura (MUR 1 km): {temperature['time'].min().date()} a {temperature['time'].max().date()} "
        f"({temperature['time'].nunique()} días, {len(temperature)} filas, {temp_path.stat().st_size / 1024:.1f} KB)"
    )
    print(
        f"Clorofila (DINEOF 2 km): {chlorophyll['time'].min().date()} a {chlorophyll['time'].max().date()} "
        f"({chlorophyll['time'].nunique()} días, {len(chlorophyll)} filas, {chl_path.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
