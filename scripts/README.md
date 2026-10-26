# Descargar datos satelitales

La aplicación lee `data/temperature.parquet` y `data/chlorophyll.parquet`. Esos archivos salen de Copernicus Marine, no de los notebooks de Google Earth Engine ni de Sentinel Hub.

El script que hay que correr es [`download_cmems.py`](download_cmems.py). Baja el mismo recorte que ya usa el mapa:

- Temperatura diaria, `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m`
- Clorofila y fitoplancton, `cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m`
- Latitud −41° a −39°. La longitud va de −74,5° (unos 75 km mar adentro de Huiro) hasta −73,42° en temperatura y hasta −73,25° en clorofila. El borde este está al menos 10 km más al este que el recorte anterior. La grilla de clorofila mide 0,25°, así que esa capa avanza una celda entera, unos 21 km. Los parquet se guardan en float32 con compresión zstd.
- Los 11 niveles entre 1,5 m y 18,5 m
- Desde el 1 de enero de 2023 hasta el día en que se ejecuta

Al terminar, reemplaza los dos parquet. Hay que reiniciar la API para que tome los archivos nuevos.

## Cuenta

Hace falta una cuenta gratuita de [Copernicus Marine](https://data.marine.copernicus.eu/register). Se inicia sesión una sola vez. Las credenciales quedan en `~/.copernicusmarine/.copernicusmarine-credentials`.

```bash
uv run --with copernicusmarine copernicusmarine login
```

## Descarga

Desde la raíz del repositorio:

```bash
uv run --with copernicusmarine --with pyarrow python scripts/download_cmems.py
```

Para cerrar la serie en otra fecha:

```bash
uv run --with copernicusmarine --with pyarrow python scripts/download_cmems.py --end 2026-09-29
```

`--start` cambia el inicio. El valor por defecto es `2023-01-01`, que es el comienzo de la serie actual.

Los NetCDF nuevos quedan en `data/`, con el rango de fechas en el nombre. Los archivos antiguos no se borran (y están ignorados por Git).
