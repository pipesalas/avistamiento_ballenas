from fastapi.testclient import TestClient

from api.main import app
from api.species import normalize_behavior, normalize_species, slugify, species_from_filename
from api.store import _safe_csv_cell

client = TestClient(app)


def test_species_and_behavior_normalization() -> None:
    assert normalize_species("Megaptera novaengliae") == "Megaptera novaeangliae"
    assert normalize_species("Chungungo (Contra felina)") == "Chungungo (Lontra felina)"
    assert normalize_species("Chungungo (Lontra felina") == "Chungungo (Lontra felina)"
    assert normalize_species("Lobo común (Otaria flavences)") == "Lobo común (Otaria flavescens)"
    assert normalize_behavior("Alimetandose ") == "Alimentándose"
    assert normalize_behavior("Forrajendo") == "Forrajeo"
    assert normalize_behavior("Navagendo") == "Navegando"
    assert slugify("Ballena azul (Balaenoptera musculus)") == "ballena-azul"
    assert species_from_filename("delfin_chileno_06_01_2023") == "Delfín chileno"
    assert species_from_filename("aleta_dorsal_fin_03_04_2023") == "Ballena fin"


def test_meta_endpoint() -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    payload = response.json()
    assert "species" in payload and len(payload["species"]) >= 13
    assert "sightings" in payload
    assert payload["sightings"]["boat_count"] > 600
    assert payload["sightings"]["shore_count"] > 500
    assert len(payload["subzones"]) == 7
    assert "ramp" in payload["environment"]["variables"][0]


def test_sightings_and_oceanographic_enrichment() -> None:
    response = client.get("/api/sightings?source=both")
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) > 1200
    props = payload["features"][0]["properties"]
    assert "sst_c" in props
    assert "chlorophyll" in props

    zones_res = client.get("/api/shore-zones")
    assert zones_res.status_code == 200
    assert len(zones_res.json()["features"]) == 7


def test_sightings_csv_export() -> None:
    response = client.get("/api/sightings.csv?source=boat")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    lines = response.text.strip().splitlines()
    assert len(lines) > 1
    assert "temperatura_superficie_c" in lines[0]
    assert _safe_csv_cell("=CMD|' /C calc'!A0") == "'=CMD|' /C calc'!A0"
    assert _safe_csv_cell("+SUM(A1:A2)") == "'+SUM(A1:A2)"
    assert _safe_csv_cell("Texto normal") == "Texto normal"


def test_timeline_and_validation() -> None:
    res_ok = client.get("/api/stats/timeline?source=both&grain=day")
    assert res_ok.status_code == 200
    assert len(res_ok.json()) > 0
    assert res_ok.headers["X-Content-Type-Options"] == "nosniff"
    assert res_ok.headers["X-Frame-Options"] == "DENY"
    assert res_ok.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    res_bad = client.get("/api/sightings?source=invalid")
    assert res_bad.status_code == 422

    res_bad_date = client.get("/api/sightings?date=not-a-date")
    assert res_bad_date.status_code == 422

    res_bad_depth = client.get("/api/environment?variable=temperature&date=2024-01-01&depth=abc")
    assert res_bad_depth.status_code == 422

    res_csv = client.get('/api/sightings.csv?species=Ballena"%0d%0aX-Injected:1')
    assert res_csv.status_code == 200
    assert res_csv.headers["Content-Disposition"] == 'attachment; filename="avistamientos_ballena-x-injected-1_historico.csv"'

    for doc_path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(doc_path).status_code == 404
