from fastapi.testclient import TestClient

from api.main import app
from api.species import normalize_species, slugify, species_from_filename

client = TestClient(app)


def test_species_normalization() -> None:
    assert normalize_species("Megaptera novaengliae") == "Megaptera novaeangliae"
    assert slugify("Ballena azul (Balaenoptera musculus)") == "ballena-azul"
    assert species_from_filename("delfin_chileno_06_01_2023") == "Delfín chileno"
    assert species_from_filename("aleta_dorsal_fin_03_04_2023") == "Ballena fin"


def test_meta_endpoint() -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    payload = response.json()
    assert "species" in payload and len(payload["species"]) > 0
    assert "sightings" in payload
    assert payload["sightings"]["boat_count"] > 0
    assert payload["sightings"]["shore_count"] > 0
    assert "ramp" in payload["environment"]["variables"][0]


def test_sightings_and_oceanographic_enrichment() -> None:
    response = client.get("/api/sightings?source=boat")
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) > 0
    props = payload["features"][0]["properties"]
    assert "sst_c" in props
    assert "chlorophyll" in props


def test_sightings_csv_export() -> None:
    response = client.get("/api/sightings.csv?source=boat")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    lines = response.text.strip().splitlines()
    assert len(lines) > 1
    assert "temperatura_superficie_c" in lines[0]


def test_timeline_and_validation() -> None:
    res_ok = client.get("/api/stats/timeline?source=both&grain=day")
    assert res_ok.status_code == 200
    assert len(res_ok.json()) > 0

    res_bad = client.get("/api/sightings?source=invalid")
    assert res_bad.status_code == 422
