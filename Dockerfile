FROM python:3.12.12-slim-bookworm AS media
WORKDIR /app
RUN pip install --no-cache-dir pillow==13.0.0
COPY scripts/prepare_photos.py scripts/prepare_photos.py
COPY data/fotos/ data/fotos/
COPY data/fotos_conteo_directo/ data/fotos_conteo_directo/
RUN python scripts/prepare_photos.py

FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY requirements-api.lock ./
RUN pip install --no-cache-dir -r requirements-api.lock
COPY api/ api/
COPY config.py ./
COPY data/*.parquet data/*.xlsx data/logo_vuelvealoceano.png data/
COPY data/tracks/ data/tracks/
COPY --from=media /app/data/fotos/ data/fotos/
COPY --from=media /app/data/fotos_conteo_directo/ data/fotos_conteo_directo/
USER 10001:10001
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\" --workers 1"]
