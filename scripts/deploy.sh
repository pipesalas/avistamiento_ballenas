#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Uso: bash scripts/deploy.sh PROJECT_ID}"
ROOT="$(git rev-parse --show-toplevel)"
REGION="us-central1"
SERVICE="avistamientos-api"
VERSION="$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/avistamientos/api:${VERSION}"

for TOOL in gcloud uv npm npx; do
  command -v "$TOOL" >/dev/null
done

# Cloud Build builds linux/amd64 even when the developer uses an ARM Mac.
gcloud builds submit "$ROOT" \
  --project "$PROJECT_ID" \
  --config "$ROOT/cloudbuild.yaml" \
  --service-account "projects/${PROJECT_ID}/serviceAccounts/avistamientos-build@${PROJECT_ID}.iam.gserviceaccount.com" \
  --substitutions "_IMAGE=${IMAGE}"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --image "$IMAGE" \
  --service-account "${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --allow-unauthenticated \
  --cpu 1 --memory 1Gi --concurrency 8 \
  --min 0 --max 2 --cpu-throttling --timeout 60

npx --yes firebase-tools@15.34.0 deploy \
  --only hosting --project "$PROJECT_ID" --config "$ROOT/firebase.json"
