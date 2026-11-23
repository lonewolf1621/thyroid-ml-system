#!/usr/bin/env bash
# Deploy to Google Cloud Run (scales to zero; fits the free tier for a demo).
# Usage: PROJECT_ID=my-project ./deploy/cloudrun.sh
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-europe-west2}"   # London
SERVICE="${SERVICE:-thyroid-ml-system}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
# Cloud Build builds the Dockerfile (including model training) from source.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars LOG_LEVEL=INFO
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
