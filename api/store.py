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
from config import zona_1, zona_2, zona_3, zona_4

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
ZONE_COLORS = {1: "#F4D03F", 2: "#E74C3C", 3: "#3498DB", 4: "#1E8449"}
ZONES = {1: zona_1, 2: zona_2, 3: zona_3, 4: zona_4}
DATE_IN_NAME = re.compile(r"(?P<d>\d{2})[_:](?P<m>\d{2})[_:](?P<y>\d{4})")


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
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d")
    return frame


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


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
        self._temp_by_day: dict[str, pd.DataFrame] = {
            str(day): group for day, group in self.temperature.groupby("day")
        }
        self._chl_by_day: dict[str, pd.DataFrame] = {
            str(day): group for day, group in self.chlorophyll.groupby("day")
        }
        self._temp_days = set(self._temp_by_day.keys())
        self._chl_days = set(self._chl_by_day.keys())

    def _load_boat(self) -> pd.DataFrame:
        frame = pd.read_excel(self.root / "data" / "datos_avistamientos.xlsx")
        frame.columns = [column.strip() for column in frame.columns]
        frame = frame[frame["Fecha"] >= "2023-01-01"].copy()
        frame["species"] = frame["Especie"].map(normalize_species)
        frame["day"] = pd.to_datetime(frame["Fecha"]).dt.strftime("%Y-%m-%d")
        frame["individuals"] = pd.to_numeric(frame["individuos"], errors="coerce").fillna(0).astype(int)
        frame["behavior"] = frame["Comportamiento"].map(normalize_behavior)
        frame["notes"] = frame["Observaciones"].map(_clean_text)
        frame["hour"] = frame["Hora"].map(_hour_text)
        frame["distance_m"] = pd.to_numeric(frame["Distancia (mts)"], errors="coerce")
        frame["latitude"] = pd.to_numeric(frame["Latitud"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["Longitud"], errors="coerce")
        frame = frame.dropna(subset=["latitude", "longitude", "species"])
        frame["source"] = "boat"
        return frame.reset_index(drop=True)

    def _load_shore(self) -> pd.DataFrame:
        frame = pd.read_excel(self.root / "data" / "Planilla conteo directo .xlsx", skiprows=1)
        frame.columns = [column.strip() for column in frame.columns]
        frame["species"] = frame["Especie"].map(normalize_species)
        frame["day"] = pd.to_datetime(frame["Fecha"]).dt.strftime("%Y-%m-%d")
        frame["individuals"] = pd.to_numeric(frame["Numero"], errors="coerce").fillna(0).astype(int)
        frame["notes"] = frame["Observaciones"].map(_clean_text)
        frame["subzone"] = pd.to_numeric(frame["Subzona"], errors="coerce").astype("Int64")
        frame["hour"] = frame["Hora"].map(_hour_text)
        frame["source"] = "shore"
        frame = frame.dropna(subset=["species", "subzone"])
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
        behaviors = sorted({value for value in self.boat["behavior"].dropna().unique()})
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
                {"id": zone_id, "color": ZONE_COLORS[zone_id]} for zone_id in (1, 2, 3, 4)
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
            else:
                ring = ZONES[int(row["subzone"])]["coordinates"][0]
                coordinates = _centroid(ring)
                subzone = int(row["subzone"])
                behavior = None
                distance_m = None
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
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

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
            day = self._temp_by_day.get(date, self.temperature.iloc[0:0])
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
            day = self._chl_by_day.get(date, self.chlorophyll.iloc[0:0])
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

    def stats_behavior(self, species: str | None = None) -> list[dict]:
        frame = self.boat.dropna(subset=["behavior"])
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
