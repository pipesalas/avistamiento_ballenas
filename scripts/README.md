# Descargar datos satelitales

La aplicación lee `data/temperature.parquet` y `data/chlorophyll.parquet`. Para maximizar la resolución espacial en la costa sin inflar el peso del repositorio ni del contenedor, se descargan únicamente las **fechas con avistamientos** (barco y orilla) dentro del recorte del área de estudio:

- **Temperatura superficial del mar (`temperature`):** **NASA JPL MUR SST v4.1** (`MUR-JPL-L4-GLOB-v4.1`), grilla diaria Level 4 libre de nubes a **0,01° (~1 km)**, leída vía Cloud-Optimized GeoTIFF (`https://data.source.coop/ausantarctic/ghrsst-mur-v2`).
- **Clorofila-a (`chlorophyll`):** **NOAA CoastWatch Multi-sensor (VIIRS SNPP + NOAA-20 + Sentinel-3A OLCI) DINEOF** (`noaacwNPPN20S3ASCIDINEOF2kmDaily` / `chloraMergedDINEOF2kmGlobalDailyWW00`), grilla diaria reconstruida (gap-filled) a **~0,02° (~2 km)**, leída desde el servidor THREDDS NCSS de NOAA CoastWatch (`https://coastwatch.noaa.gov/thredds/ncss/grid/chloraMergedDINEOF2kmGlobalDailyWW00`).
- **Bounding box por defecto:** longitud `−74,15°` a `−73,35°` y latitud `−40,30°` a `−39,60°` (cubre todas las subzonas de orilla 1–7, todos los tracks de navegación y avistamientos con ~35 km de margen mar adentro y norte-sur).
- ** Compresión:** `float32` con `zstd` nivel 9 (~1,7 MB para temperatura y ~1,6 MB para clorofila).

No requiere cuentas ni credenciales.

## Descarga

Desde la raíz del repositorio:

```bash
uv run --with-requirements requirements-api.txt --with rasterio --with scipy --with xarray python scripts/download_cmems.py
```

Para ajustar el recorte espacial (`LON_MIN LAT_MIN LON_MAX LAT_MAX`):

```bash
uv run --with-requirements requirements-api.txt --with rasterio --with scipy --with xarray python scripts/download_cmems.py --bbox -74.15 -40.30 -73.35 -39.60
```
