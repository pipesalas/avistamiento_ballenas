#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:?Uso: bash scripts/setup_gcp.sh PROJECT_ID [GITHUB_REPO]}"
GITHUB_REPO="${2:-pipesalas/avistamiento_ballenas}"
REGION="us-central1"
RUNTIME_ACCOUNT="avistamientos-api@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_ACCOUNT="avistamientos-build@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_ACCOUNT="github-deploy@${PROJECT_ID}.iam.gserviceaccount.com"

# Run after creating the Firebase project and linking its billing account.
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  logging.googleapis.com \
  firebasehosting.googleapis.com \
  --project "$PROJECT_ID"

if ! gcloud artifacts repositories describe avistamientos --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create avistamientos \
    --repository-format docker --location "$REGION" --project "$PROJECT_ID"
fi

CLEANUP_POLICY_FILE="$(mktemp)"
cat > "$CLEANUP_POLICY_FILE" <<'EOF'
[
  {
    "name": "keep-recent-5",
    "action": {"type": "Keep"},
    "mostRecentVersions": {"keepCount": 5}
  },
  {
    "name": "delete-older-than-7d",
    "action": {"type": "Delete"},
    "condition": {"olderThan": "604800s"}
  }
]
EOF
gcloud artifacts repositories set-cleanup-policies avistamientos \
  --location "$REGION" \
  --project "$PROJECT_ID" \
  --policy "$CLEANUP_POLICY_FILE" \
  --no-dry-run >/dev/null
rm -f "$CLEANUP_POLICY_FILE"

for ACCOUNT in avistamientos-api avistamientos-build github-deploy; do
  if ! gcloud iam service-accounts describe "${ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$ACCOUNT" --project "$PROJECT_ID"
  fi
done

# The API reads bundled files and needs no project IAM roles.
# The build account reads the uploaded source, pushes images and writes logs.
for ROLE in roles/artifactregistry.writer roles/storage.objectViewer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${BUILD_ACCOUNT}" --role "$ROLE" --condition=None >/dev/null
done

# The GitHub Actions deploy account pushes images, deploys Cloud Run and releases Firebase Hosting.
for ROLE in \
  roles/artifactregistry.writer \
  roles/run.admin \
  roles/firebasehosting.admin \
  roles/cloudbuild.builds.editor \
  roles/storage.objectAdmin \
  roles/serviceusage.serviceUsageConsumer \
  roles/viewer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${DEPLOY_ACCOUNT}" --role "$ROLE" --condition=None >/dev/null
done

for TARGET_SA in "$RUNTIME_ACCOUNT" "$BUILD_ACCOUNT"; do
  gcloud iam service-accounts add-iam-policy-binding "$TARGET_SA" \
    --member "serviceAccount:${DEPLOY_ACCOUNT}" \
    --role "roles/iam.serviceAccountUser" \
    --project "$PROJECT_ID" >/dev/null
done

if ! gcloud iam workload-identity-pools describe github --location=global --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create github \
    --location="global" \
    --display-name="GitHub Actions Pool" \
    --project "$PROJECT_ID"
fi

OIDC_CONDITION="assertion.repository == '${GITHUB_REPO}' && assertion.ref == 'refs/heads/main'"
if ! gcloud iam workload-identity-pools providers describe github-provider --workload-identity-pool=github --location=global --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc github-provider \
    --location="global" \
    --workload-identity-pool="github" \
    --display-name="GitHub OIDC Provider" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="$OIDC_CONDITION" \
    --project "$PROJECT_ID"
else
  gcloud iam workload-identity-pools providers update-oidc github-provider \
    --location="global" \
    --workload-identity-pool="github" \
    --attribute-condition="$OIDC_CONDITION" \
    --project "$PROJECT_ID" >/dev/null
fi

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_ACCOUNT" \
  --project "$PROJECT_ID" \
  --role "roles/iam.workloadIdentityUser" \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}" >/dev/null

printf 'Proyecto preparado. Cuenta de ejecución: %s\n' "$RUNTIME_ACCOUNT"
printf 'Proveedor OIDC: projects/%s/locations/global/workloadIdentityPools/github/providers/github-provider\n' "$PROJECT_NUMBER"
printf 'Cuenta de despliegue GitHub: %s\n' "$DEPLOY_ACCOUNT"
