"""Download the same Copernicus Marine subsets the app uses, through today.

Temperature: cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m
Chlorophyll and phytoplankton: cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m

Requires a free Copernicus Marine account. Log in once with:
    uv run --with copernicusmarine copernicusmarine login
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import copernicusmarine
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
START = "2023-01-01"

# Eastern edge is 10 km east of the previous box. At 41°S that is about 0.12°
# of longitude. Each product snaps to its own grid, so the bound sits just
# east of the first cell that covers those 10 km: two steps of 1/12° for
# temperature (~14 km) and one step of 0.25° for chlorophyll (~21 km).
# The western edge sits at -74.5°, about 75 km offshore of Huiro. Farther
# west is open ocean and made the temperature file too large to store.
WEST_LONGITUDE = -74.5
TEMPERATURE = {
    "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    "variables": ["thetao"],
    "minimum_longitude": WEST_LONGITUDE,
    "maximum_longitude": -73.41,
    "minimum_latitude": -41.0,
    "maximum_latitude": -39.0,
    "minimum_depth": 1.54,
    "maximum_depth": 18.50,
}
CHLOROPHYLL = {
    "dataset_id": "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    "variables": ["chl", "phyc"],
    "minimum_longitude": WEST_LONGITUDE,
    "maximum_longitude": -73.24,
    "minimum_latitude": -41.0,
    "maximum_latitude": -39.0,
    "minimum_depth": 1.54,
    "maximum_depth": 18.50,
}


def download(spec: dict, start: str, end: str) -> Path:
    stamp = f"{start.replace('-', '')}_{end.replace('-', '')}"
    filename = f"{spec['dataset_id']}_{stamp}.nc"
    print(f"Descargando {spec['dataset_id']} ({start} a {end})", flush=True)
    result = copernicusmarine.subset(
        dataset_id=spec["dataset_id"],
        variables=spec["variables"],
        minimum_longitude=spec["minimum_longitude"],
        maximum_longitude=spec["maximum_longitude"],
        minimum_latitude=spec["minimum_latitude"],
        maximum_latitude=spec["maximum_latitude"],
        minimum_depth=spec["minimum_depth"],
        maximum_depth=spec["maximum_depth"],
        start_datetime=f"{start}T00:00:00",
        end_datetime=f"{end}T23:59:59",
        output_directory=str(DATA),
        output_filename=filename,
        overwrite=True,
    )
    return Path(result.file_path)


def to_frame(path: Path, rename: dict[str, str], value_column: str) -> pd.DataFrame:
    dataset = xr.open_dataset(path)
    frame = dataset[list(rename)].to_dataframe().reset_index().rename(columns=rename)
    dataset.close()
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(None)
    keep = ["time", "depth", "latitude", "longitude", *rename.values()]
    frame = frame.loc[frame[value_column] > -100, keep].copy()
    for column in ("depth", "latitude", "longitude", *rename.values()):
        frame[column] = frame[column].astype("float32")
    return frame.sort_values(["time", "depth", "latitude", "longitude"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga temperatura, clorofila y fitoplancton hasta una fecha.")
    parser.add_argument("--start", default=START, help="Fecha inicial YYYY-MM-DD. Por defecto 2023-01-01.")
    parser.add_argument("--end", default=date.today().isoformat(), help="Fecha final YYYY-MM-DD. Por defecto hoy.")
    args = parser.parse_args()

    temperature_nc = download(TEMPERATURE, args.start, args.end)
    chlorophyll_nc = download(CHLOROPHYLL, args.start, args.end)

    temperature = to_frame(temperature_nc, {"thetao": "temperature"}, "temperature")
    chlorophyll = to_frame(chlorophyll_nc, {"chl": "chlorophyll", "phyc": "phyc"}, "chlorophyll")
    parquet_options = {"index": False, "compression": "zstd", "compression_level": 9}
    temperature.to_parquet(DATA / "temperature.parquet", **parquet_options)
    chlorophyll.to_parquet(DATA / "chlorophyll.parquet", **parquet_options)

    print(
        f"Temperatura: {temperature['time'].min().date()} a {temperature['time'].max().date()} "
        f"({temperature['time'].nunique()} días, {len(temperature)} filas)"
    )
    print(
        f"Clorofila: {chlorophyll['time'].min().date()} a {chlorophyll['time'].max().date()} "
        f"({chlorophyll['time'].nunique()} días, {len(chlorophyll)} filas)"
    )


if __name__ == "__main__":
    main()
