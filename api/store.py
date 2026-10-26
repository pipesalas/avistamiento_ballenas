"""Load sightings, tracks, photos and environmental grids once, then serve one day."""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import gpxpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.species import (
    PALETTE,
    normalize_behavior,
    normalize_species,
    short_name,
    slugify,
    species_from_filename,
)
from config import zona_1, zona_2, zona_3, zona_4, zona_5, zona_6, zona_7

TEMP_COLORS = [
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
]
CHL_COLORS = [
    [237, 248, 229],
    [199, 233, 192],
    [161, 217, 155],
    [116, 196, 118],
    [65, 171, 93],
    [35, 139, 69],
    [0, 109, 44],
    [0, 68, 27],
]
ZONE_COLORS = {
    1: "#F4D03F",
    2: "#E74C3C",
    3: "#3498DB",
    4: "#1E8449",
    5: "#9B59B6",
    6: "#E67E22",
    7: "#16A085",
}
ZONES = {1: zona_1, 2: zona_2, 3: zona_3, 4: zona_4, 5: zona_5, 6: zona_6, 7: zona_7}
DATE_IN_NAME = re.compile(r"(?P<d>\d{2})[_:](?P<m>\d{2})[_:](?P<y>\d{4})")


def _utm18s_to_lonlat(easting: float, northing: float) -> tuple[float, float] | None:
    if pd.isna(easting) or pd.isna(northing):
        return None
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e = float(np.sqrt(2 * f - f * f))
    e1sq = e * e / (1 - e * e)
    x = float(easting) - 500000.0
    y = float(northing) - 10000000.0
    m_val = y / k0
    mu = m_val / (a * (1 - e * e / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - float(np.sqrt(1 - e * e))) / (1 + float(np.sqrt(1 - e * e)))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = (
        mu
        + j1 * float(np.sin(2 * mu))
        + j2 * float(np.sin(4 * mu))
        + j3 * float(np.sin(6 * mu))
        + j4 * float(np.sin(8 * mu))
    )
    c1 = e1sq * float(np.cos(fp)) ** 2
    t1 = float(np.tan(fp)) ** 2
    r1 = a * (1 - e * e) / (1 - e * e * float(np.sin(fp)) ** 2) ** 1.5
    n1 = a / float(np.sqrt(1 - e * e * float(np.sin(fp)) ** 2))
    d_val = x / (n1 * k0)
    q1 = n1 * float(np.tan(fp)) / r1
    q2 = d_val**2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e1sq) * d_val**4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 3 * c1**2 - 252 * e1sq) * d_val**6 / 720
    lat = float(np.degrees(fp - q1 * (q2 - q3 + q4)))
    q5 = d_val
    q6 = (1 + 2 * t1 + c1) * d_val**3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e1sq + 24 * t1**2) * d_val**5 / 120
    lon = -75.0 + float(np.degrees((q5 - q6 + q7) / float(np.cos(fp))))
    if not (-40.3 <= lat <= -39.6 and -74.0 <= lon <= -73.2):
        return None
    return round(lon, 5), round(lat, 5)


def _hex(rgb: list[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _color_for_value(value: float, vmin: float, vmax: float, ramp: list[list[int]]) -> str:
    if vmin == vmax:
        return _hex(ramp[-1])
    bins = np.linspace(vmin, vmax, len(ramp))
    for index, edge in enumerate(bins):
        if value < edge:
            return _hex(ramp[index])
    return _hex(ramp[-1])


def _cell_ring(lon: float, lat: float, size: float) -> list[list[float]]:
    half = size / 2
    return [
        [round(lon - half, 5), round(lat - half, 5)],
        [round(lon + half, 5), round(lat - half, 5)],
        [round(lon + half, 5), round(lat + half, 5)],
        [round(lon - half, 5), round(lat + half, 5)],
        [round(lon - half, 5), round(lat - half, 5)],
    ]


def _centroid(ring: list) -> list[float]:
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return [round(lon, 5), round(lat, 5)]


def _day_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare a newly loaded grid, formatting each distinct day only once."""
    if not frame["time"].is_monotonic_increasing:
        frame = frame.sort_values("time", kind="stable")
    codes, days = pd.factorize(pd.to_datetime(frame["time"]).dt.normalize(), sort=True)
    frame["day"] = pd.Categorical.from_codes(codes, categories=days.strftime("%Y-%m-%d"))
    return frame


def _day_slices(frame: pd.DataFrame) -> dict[str, slice]:
    """Index contiguous days without retaining a second copy of the grid."""
    codes = frame["day"].cat.codes.to_numpy()
    if not len(codes):
        return {}
    boundaries = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    categories = frame["day"].cat.categories
    return {
        str(categories[codes[start]]): slice(int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if codes[start] >= 0
    }


def _day_frame(frame: pd.DataFrame, days: dict[str, slice], date: str) -> pd.DataFrame:
    return frame.iloc[days.get(date, slice(0, 0))]


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_csv_cell(value: str | None) -> str:
    if not value:
        return ""
    if value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{value}"
    return value


def _hour_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M")
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if " " in text and ":" in text:
        return text.split(" ")[-1][:5]
    return text[:5] if len(text) >= 5 and text[2] == ":" else text


class DataStore:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.boat = self._load_boat()
        self.shore = self._load_shore()
        self._tracks = self._load_tracks()
        self._photos = self._load_photos()
        self.temperature = _day_column(
            pd.read_parquet(root / "data" / "temperature.parquet")
        )
        self.chlorophyll = _day_column(
            pd.read_parquet(root / "data" / "chlorophyll.parquet")
        )
        self.min_depth = float(self.temperature["depth"].min())
        self.depths = sorted(float(value) for value in self.temperature["depth"].unique())
        self.species = self._build_species()
        self.by_slug = {item["slug"]: item for item in self.species}
        self._temp_by_day = _day_slices(self.temperature)
        self._chl_by_day = _day_slices(self.chlorophyll)
        self._temp_days = set(self._temp_by_day.keys())
        self._chl_days = set(self._chl_by_day.keys())
        self._enrich_environment(self.boat)
        self._enrich_environment(self.shore)

    def _enrich_environment(self, frame: pd.DataFrame) -> None:
        n = len(frame)
        sst_values: list[float | None] = [None] * n
        chl_values: list[float | None] = [None] * n
        if n == 0:
            frame["sst_c"] = sst_values
            frame["chlorophyll"] = chl_values
            return

        lats = frame["latitude"].to_numpy(dtype=float, na_value=np.nan)
        lons = frame["longitude"].to_numpy(dtype=float, na_value=np.nan)
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))

        for day, indices in frame.groupby("day", sort=False).groups.items():
            idx = np.asarray(indices, dtype=int)
            idx = idx[valid_mask[idx]]
            if len(idx) == 0:
                continue
            pts_lat = lats[idx][:, None]
            pts_lon = lons[idx][:, None]
            day_str = str(day)

            temp_day = _day_frame(self.temperature, self._temp_by_day, day_str)
            if not temp_day.empty:
                surface = temp_day[np.isclose(temp_day["depth"], self.min_depth)].dropna(subset=["temperature"])
                if not surface.empty:
                    grid_lat = surface["latitude"].to_numpy(dtype=float)[None, :]
                    grid_lon = surface["longitude"].to_numpy(dtype=float)[None, :]
                    grid_val = surface["temperature"].to_numpy(dtype=float)
                    nearest_idx = ((pts_lat - grid_lat) ** 2 + (pts_lon - grid_lon) ** 2).argmin(axis=1)
                    for pos, val in zip(idx, grid_val[nearest_idx]):
                        sst_values[int(pos)] = round(float(val), 1)

            chl_day = _day_frame(self.chlorophyll, self._chl_by_day, day_str)
            if not chl_day.empty:
                valid_chl = chl_day.dropna(subset=["chlorophyll"])
                if not valid_chl.empty:
                    grid_lat = valid_chl["latitude"].to_numpy(dtype=float)[None, :]
                    grid_lon = valid_chl["longitude"].to_numpy(dtype=float)[None, :]
                    grid_val = valid_chl["chlorophyll"].to_numpy(dtype=float)
                    nearest_idx = ((pts_lat - grid_lat) ** 2 + (pts_lon - grid_lon) ** 2).argmin(axis=1)
                    for pos, val in zip(idx, grid_val[nearest_idx]):
                        chl_values[int(pos)] = round(float(val), 2)

        frame["sst_c"] = sst_values
        frame["chlorophyll"] = chl_values

    def _workbook_path(self) -> Path:
        matches = sorted(self.root.joinpath("data").glob("Monitoreo*.xlsx"))
        if not matches:
            raise FileNotFoundError("No se encontró el archivo Excel de monitoreo en data/")
        return matches[0]

    def _load_boat(self) -> pd.DataFrame:
        wb = self._workbook_path()
        frame = pd.read_excel(wb, sheet_name="Muestreo a distancia")
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.dropna(subset=["Fecha", "Especie"]).copy()
        frame["day"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame[frame["day"] >= "2023-01-01"].copy()
        frame["species"] = frame["Especie"].map(normalize_species)
        ind_col = "# individuos" if "# individuos" in frame.columns else "individuos"
        frame["individuals"] = pd.to_numeric(frame[ind_col], errors="coerce").fillna(0).astype(int)
        frame["behavior"] = frame["Comportamiento"].map(normalize_behavior)
        frame["notes"] = frame["Observaciones"].map(_clean_text)
        frame["hour"] = frame["Hora"].map(_hour_text)
        frame["distance_m"] = pd.to_numeric(frame["Distancia (mts)"], errors="coerce")
        frame["latitude"] = pd.to_numeric(frame["Latitud"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["Longitud"], errors="coerce")
        frame = frame.dropna(subset=["latitude", "longitude", "species", "day"])
        frame["source"] = "boat"
        return frame.reset_index(drop=True)

    def _load_shore(self) -> pd.DataFrame:
        wb = self._workbook_path()
        with pd.ExcelFile(wb) as xl:
            shore_sheet = next(
                (name for name in xl.sheet_names if "conteo directo" in name.lower()),
                xl.sheet_names[-1],
            )
            frame = pd.read_excel(xl, sheet_name=shore_sheet)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.dropna(subset=["Fecha", "Especie", "Subzona"]).copy()
        frame["species"] = frame["Especie"].map(normalize_species)
        frame["day"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["individuals"] = pd.to_numeric(frame["Numero"], errors="coerce").fillna(0).astype(int)
        frame["notes"] = frame["Observaciones"].map(_clean_text)
        frame["subzone"] = pd.to_numeric(frame["Subzona"], errors="coerce").astype("Int64")
        frame["hour"] = frame["Hora"].map(_hour_text)
        if "Comportamiento" in frame.columns:
            frame["behavior"] = frame["Comportamiento"].map(normalize_behavior)
        else:
            frame["behavior"] = None

        easting_col = next((c for c in frame.columns if "este" in c.lower()), None)
        northing_col = next((c for c in frame.columns if "norte" in c.lower()), None)
        lons: list[float | None] = []
        lats: list[float | None] = []
        for idx, row in frame.iterrows():
            coord = None
            if easting_col and northing_col:
                e_val = pd.to_numeric(row[easting_col], errors="coerce")
                n_val = pd.to_numeric(row[northing_col], errors="coerce")
                coord = _utm18s_to_lonlat(e_val, n_val)
            if coord is None:
                sz = int(row["subzone"]) if pd.notna(row["subzone"]) and int(row["subzone"]) in ZONES else 1
                c_lon, c_lat = _centroid(ZONES[sz]["coordinates"][0])
                coord = (c_lon, c_lat)
            lons.append(coord[0])
            lats.append(coord[1])
        frame["longitude"] = lons
        frame["latitude"] = lats
        frame["source"] = "shore"
        frame = frame.dropna(subset=["species", "subzone", "day"])
        return frame.reset_index(drop=True)

    def _load_tracks(self) -> list[dict]:
        tracks = []
        for path in sorted((self.root / "data" / "tracks").glob("*.gpx")):
            match = DATE_IN_NAME.search(path.stem)
            if not match:
                continue
            date = f"{match.group('y')}-{match.group('m')}-{match.group('d')}"
            with path.open() as handle:
                gpx = gpxpy.parse(handle)
            coordinates = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coordinates.append([round(point.longitude, 5), round(point.latitude, 5)])
            if len(coordinates) < 2:
                continue
            tracks.append({"date": date, "coordinates": coordinates})
        return tracks

    def _load_photos(self) -> list[dict]:
        photos = []
        folders = {
            "boat": self.root / "data" / "fotos",
            "shore": self.root / "data" / "fotos_conteo_directo",
        }
        for source, folder in folders.items():
            mount = "fotos" if source == "boat" else "fotos_conteo_directo"
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                match = DATE_IN_NAME.search(path.name)
                if not match:
                    continue
                date = f"{match.group('y')}-{match.group('m')}-{match.group('d')}"
                hint = species_from_filename(path.stem)
                photos.append(
                    {
                        "source": source,
                        "date": date,
                        "file": path.name,
                        "url": f"/media/{mount}/{quote(path.name)}",
                        "species_short": hint,
                        "caption": path.stem.replace("_", " ").replace(":", " "),
                    }
                )
                thumbnail = folder / "thumbnails" / f"{path.name}.webp"
                if thumbnail.is_file():
                    photos[-1]["thumbnail_url"] = f"/media/{mount}/thumbnails/{quote(thumbnail.name)}"
        return photos

    def _build_species(self) -> list[dict]:
        names = sorted(set(self.boat["species"]) | set(self.shore["species"]), key=slugify)
        catalog = []
        for index, name in enumerate(names):
            catalog.append(
                {
                    "name": name,
                    "short": short_name(name),
                    "slug": slugify(name),
                    "color": PALETTE[index % len(PALETTE)],
                }
            )
        return catalog

    def _species_record(self, name: str) -> dict:
        slug = slugify(name)
        return self.by_slug.get(
            slug,
            {"name": name, "short": short_name(name), "slug": slug, "color": "#667085"},
        )

    def meta(self) -> dict:
        behaviors = sorted(
            {value for value in pd.concat([self.boat["behavior"], self.shore["behavior"]]).dropna().unique()}
        )
        sighting_days = sorted(set(self.boat["day"]) | set(self.shore["day"]))
        env_days = sorted(self._temp_days | self._chl_days)
        temp_ramp = [_hex(rgb) for rgb in TEMP_COLORS]
        chl_ramp = [_hex(rgb) for rgb in CHL_COLORS]
        return {
            "project": {
                "title": "Monitoreo de mamíferos marinos en las localidades de Huiro y Chaihuín",
                "caption": "Proyecto financiado por TNC Chile, GORE Los Ríos y ONG Vuelve Al Océano.",
            },
            "species": self.species,
            "behaviors": behaviors,
            "subzones": [
                {"id": zone_id, "color": ZONE_COLORS[zone_id]} for zone_id in sorted(ZONES.keys())
            ],
            "sightings": {
                "start": sighting_days[0] if sighting_days else None,
                "end": sighting_days[-1] if sighting_days else None,
                "dates": sighting_days,
                "boat_dates": sorted(set(self.boat["day"])),
                "shore_dates": sorted(set(self.shore["day"])),
                "boat_count": int(len(self.boat)),
                "shore_count": int(len(self.shore)),
                "boat_individuals": int(self.boat["individuals"].sum()),
                "shore_individuals": int(self.shore["individuals"].sum()),
            },
            "environment": {
                "start": env_days[0] if env_days else None,
                "end": env_days[-1] if env_days else None,
                "depths": [round(depth, 3) for depth in self.depths],
                "surface_depth": round(self.min_depth, 3),
                "variables": [
                    {"id": "temperature", "label": "Temperatura", "unit": "°C", "ramp": temp_ramp},
                    {"id": "chlorophyll", "label": "Clorofila", "unit": "mg/m³", "ramp": chl_ramp},
                    {"id": "phyc", "label": "Fitoplancton", "unit": "mmol/m³", "ramp": chl_ramp},
                ],
            },
        }

    def _filter_sightings(
        self,
        frame: pd.DataFrame,
        date: str | None,
        date_from: str | None,
        date_to: str | None,
        species: str | None,
        behavior: str | None,
    ) -> pd.DataFrame:
        selected = frame
        if date:
            selected = selected[selected["day"] == date]
        if date_from:
            selected = selected[selected["day"] >= date_from]
        if date_to:
            selected = selected[selected["day"] <= date_to]
        if species:
            record = self.by_slug.get(species)
            target = record["name"] if record else species
            selected = selected[selected["species"] == target]
        if behavior and "behavior" in selected.columns:
            selected = selected[selected["behavior"] == behavior]
        return selected

    def sightings(
        self,
        source: str = "both",
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        species: str | None = None,
        behavior: str | None = None,
    ) -> dict:
        frames = []
        if source in ("boat", "both"):
            frames.append(self.boat)
        if source in ("shore", "both"):
            frames.append(self.shore)
        if not frames:
            return {"type": "FeatureCollection", "features": []}
        selected = pd.concat(frames, ignore_index=True)
        selected = self._filter_sightings(selected, date, date_from, date_to, species, behavior)
        features = []
        for row in selected.to_dict(orient="records"):
            record = self._species_record(row["species"])
            if row["source"] == "boat":
                coordinates = [round(float(row["longitude"]), 5), round(float(row["latitude"]), 5)]
                subzone = None
                behavior = row.get("behavior")
                distance = row.get("distance_m")
                distance_m = None if distance is None or pd.isna(distance) else float(distance)
                sst = row.get("sst_c")
                sst_c = None if sst is None or pd.isna(sst) else float(sst)
                chl = row.get("chlorophyll")
                chlorophyll = None if chl is None or pd.isna(chl) else float(chl)
            else:
                subzone = int(row["subzone"])
                if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
                    coordinates = [round(float(row["longitude"]), 5), round(float(row["latitude"]), 5)]
                else:
                    ring = ZONES[subzone]["coordinates"][0]
                    coordinates = _centroid(ring)
                behavior = row.get("behavior")
                distance_m = None
                sst = row.get("sst_c")
                sst_c = None if sst is None or pd.isna(sst) else float(sst)
                chl = row.get("chlorophyll")
                chlorophyll = None if chl is None or pd.isna(chl) else float(chl)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coordinates},
                    "properties": {
                        "source": row["source"],
                        "date": row["day"],
                        "species": record["name"],
                        "species_short": record["short"],
                        "slug": record["slug"],
                        "color": record["color"],
                        "individuals": int(row["individuals"]),
                        "behavior": None if behavior is None or (isinstance(behavior, float) and np.isnan(behavior)) else behavior,
                        "notes": row["notes"],
                        "hour": row["hour"],
                        "distance_m": distance_m,
                        "subzone": subzone,
                        "sst_c": sst_c,
                        "chlorophyll": chlorophyll,
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def sightings_csv(
        self,
        source: str = "both",
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        species: str | None = None,
        behavior: str | None = None,
    ) -> str:
        collection = self.sightings(source, date, date_from, date_to, species, behavior)
        rows = []
        for feature in collection["features"]:
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            rows.append(
                {
                    "fecha": props["date"],
                    "hora": props["hour"],
                    "fuente": props["source"],
                    "especie": _safe_csv_cell(props["species"]),
                    "especie_corto": _safe_csv_cell(props["species_short"]),
                    "individuos": props["individuals"],
                    "comportamiento": _safe_csv_cell(props["behavior"]),
                    "subzona": props["subzone"] or "",
                    "longitud": coords[0],
                    "latitud": coords[1],
                    "temperatura_superficie_c": "" if props["sst_c"] is None else props["sst_c"],
                    "clorofila_mg_m3": "" if props["chlorophyll"] is None else props["chlorophyll"],
                    "observaciones": _safe_csv_cell(props["notes"]),
                }
            )
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def shore_zones(
        self,
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        species: str | None = None,
    ) -> dict:
        selected = self._filter_sightings(self.shore, date, date_from, date_to, species, None)
        features = []
        if selected.empty:
            return {"type": "FeatureCollection", "features": features}
        for subzone, group in selected.groupby("subzone"):
            zone_id = int(subzone)
            lines = []
            for species_name, species_group in group.groupby("species"):
                record = self._species_record(species_name)
                lines.append(f"{record['short']}: {int(species_group['individuals'].sum())} ejemplares")
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": ZONES[zone_id]["coordinates"],
                    },
                    "properties": {
                        "subzone": zone_id,
                        "color": ZONE_COLORS[zone_id],
                        "events": int(len(group)),
                        "individuals": int(group["individuals"].sum()),
                        "summary": "\n".join(lines),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def tracks(self, date: str | None = None) -> dict:
        chosen = self._tracks if not date else [track for track in self._tracks if track["date"] == date]
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": track["coordinates"]},
                    "properties": {"date": track["date"]},
                }
                for track in chosen
            ],
        }

    def environment(self, variable: str, date: str, depth: str = "surface") -> dict:
        if variable == "temperature":
            day = _day_frame(self.temperature, self._temp_by_day, date)
            unit = "°C"
            ramp = TEMP_COLORS
            grid = 0.08
            if depth == "mean":
                grouped = day.groupby(["latitude", "longitude"], as_index=False)["temperature"].mean()
                column = "temperature"
            elif depth == "surface":
                grouped = day[np.isclose(day["depth"], self.min_depth)]
                column = "temperature"
            else:
                grouped = day[np.isclose(day["depth"].astype(float), float(depth), atol=0.02)]
                column = "temperature"
        elif variable in ("chlorophyll", "phyc"):
            day = _day_frame(self.chlorophyll, self._chl_by_day, date)
            grouped = day.groupby(["latitude", "longitude"], as_index=False)[[variable]].mean()
            column = variable
            unit = "mg/m³" if variable == "chlorophyll" else "mmol/m³"
            ramp = CHL_COLORS
            grid = 0.2
        else:
            raise ValueError(variable)

        values = grouped[column].dropna() if not grouped.empty else pd.Series(dtype=float)
        vmin = float(values.min()) if len(values) else None
        vmax = float(values.max()) if len(values) else None
        features = []
        for row in grouped.itertuples(index=False):
            value = getattr(row, column)
            if pd.isna(value):
                continue
            rounded = round(float(value), 1)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_cell_ring(float(row.longitude), float(row.latitude), grid)],
                    },
                    "properties": {
                        "value": rounded,
                        "label": f"{rounded} {unit}",
                        "color": _color_for_value(rounded, vmin, vmax, ramp) if vmin is not None else _hex(ramp[-1]),
                    },
                }
            )
        return {
            "type": "FeatureCollection",
            "properties": {
                "variable": variable,
                "date": date,
                "depth": depth if variable == "temperature" else "mean",
                "unit": unit,
                "min": None if vmin is None else round(vmin, 1),
                "max": None if vmax is None else round(vmax, 1),
            },
            "features": features,
        }

    def photos(
        self,
        source: str = "both",
        date: str | None = None,
        species: str | None = None,
    ) -> list[dict]:
        record = self.by_slug.get(species) if species else None
        short = record["short"] if record else None
        results = []
        for photo in self._photos:
            if source != "both" and photo["source"] != source:
                continue
            if date and photo["date"] != date:
                continue
            if short and photo["species_short"] != short:
                continue
            species_record = None
            if photo["species_short"]:
                matches = [item for item in self.species if item["short"] == photo["species_short"]]
                species_record = matches[0] if matches else None
            results.append(
                {
                    "url": photo["url"],
                    "date": photo["date"],
                    "source": photo["source"],
                    "caption": photo["caption"],
                    "species": None if species_record is None else species_record["name"],
                    "species_short": photo["species_short"],
                    "slug": None if species_record is None else species_record["slug"],
                }
            )
            if "thumbnail_url" in photo:
                results[-1]["thumbnail_url"] = photo["thumbnail_url"]
        return results

    def _combined(self, source: str) -> pd.DataFrame:
        frames = []
        if source in ("boat", "both"):
            frames.append(self.boat.assign(source="boat"))
        if source in ("shore", "both"):
            frames.append(self.shore.assign(source="shore"))
        if not frames:
            return pd.DataFrame(columns=["day", "species", "individuals", "source", "behavior"])
        return pd.concat(frames, ignore_index=True)

    def stats_species(self, source: str = "both") -> list[dict]:
        frame = self._combined(source)
        rows = []
        for item in self.species:
            subset = frame[frame["species"] == item["name"]]
            boat = subset[subset["source"] == "boat"]
            shore = subset[subset["source"] == "shore"]
            if subset.empty and source != "both":
                continue
            rows.append(
                {
                    **item,
                    "boat_events": int(len(boat)),
                    "boat_individuals": int(boat["individuals"].sum()) if len(boat) else 0,
                    "shore_events": int(len(shore)),
                    "shore_individuals": int(shore["individuals"].sum()) if len(shore) else 0,
                }
            )
        if source == "boat":
            rows = [row for row in rows if row["boat_events"]]
        elif source == "shore":
            rows = [row for row in rows if row["shore_events"]]
        else:
            rows = [row for row in rows if row["boat_events"] or row["shore_events"]]
        return rows

    def stats_timeline(self, source: str = "both", species: str | None = None, grain: str = "day") -> list[dict]:
        frame = self._filter_sightings(self._combined(source), None, None, None, species, None)
        if frame.empty:
            return []
        frame = frame.copy()
        if grain == "month":
            frame["bucket"] = frame["day"].str.slice(0, 7)
        else:
            frame["bucket"] = frame["day"]
        grouped = frame.groupby(["bucket", "source"], as_index=False).agg(
            events=("individuals", "size"),
            individuals=("individuals", "sum"),
        )
        return [
            {
                "date": row.bucket,
                "source": row.source,
                "events": int(row.events),
                "individuals": int(row.individuals),
            }
            for row in grouped.itertuples(index=False)
        ]

    def stats_seasonality(self, source: str = "both", species: str | None = None) -> list[dict]:
        frame = self._filter_sightings(self._combined(source), None, None, None, species, None)
        if frame.empty:
            return []
        frame = frame.copy()
        frame["year"] = frame["day"].str.slice(0, 4).astype(int)
        frame["month"] = frame["day"].str.slice(5, 7).astype(int)
        grouped = frame.groupby(["year", "month", "source"], as_index=False).agg(
            events=("individuals", "size"),
            individuals=("individuals", "sum"),
        )
        return [
            {
                "year": int(row.year),
                "month": int(row.month),
                "source": row.source,
                "events": int(row.events),
                "individuals": int(row.individuals),
            }
            for row in grouped.itertuples(index=False)
        ]

    def stats_behavior(self, species: str | None = None, source: str = "both") -> list[dict]:
        frame = self._combined(source).dropna(subset=["behavior"])
        if species:
            record = self.by_slug.get(species)
            if record:
                frame = frame[frame["species"] == record["name"]]
        if frame.empty:
            return []
        grouped = frame.groupby(["species", "behavior"], as_index=False).agg(
            events=("individuals", "size"),
            individuals=("individuals", "sum"),
        )
        rows = []
        for row in grouped.itertuples(index=False):
            record = self._species_record(row.species)
            rows.append(
                {
                    "species": record["name"],
                    "species_short": record["short"],
                    "slug": record["slug"],
                    "color": record["color"],
                    "behavior": row.behavior,
                    "events": int(row.events),
                    "individuals": int(row.individuals),
                }
            )
        return rows


@lru_cache(maxsize=1)
def get_store() -> DataStore:
    return DataStore()
