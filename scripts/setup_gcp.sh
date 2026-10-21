#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Uso: bash scripts/setup_gcp.sh PROJECT_ID}"
REGION="us-central1"
RUNTIME_ACCOUNT="avistamientos-api@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_ACCOUNT="avistamientos-build@${PROJECT_ID}.iam.gserviceaccount.com"

# Run after creating the Firebase project and linking its billing account.
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  firebasehosting.googleapis.com \
  --project "$PROJECT_ID"

if ! gcloud artifacts repositories describe avistamientos --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create avistamientos \
    --repository-format docker --location "$REGION" --project "$PROJECT_ID"
fi

for ACCOUNT in avistamientos-api avistamientos-build; do
  if ! gcloud iam service-accounts describe "${ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$ACCOUNT" --project "$PROJECT_ID"
  fi
done

# The API reads bundled files and needs no project IAM roles.
# The build account reads the uploaded source, pushes images and writes logs.
for ROLE in roles/artifactregistry.writer roles/storage.objectViewer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${BUILD_ACCOUNT}" --role "$ROLE" --condition=None
done

printf 'Proyecto preparado. Cuenta de ejecución: %s\n' "$RUNTIME_ACCOUNT"
