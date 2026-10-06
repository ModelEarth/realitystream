#!/usr/bin/env bash
# Deploy RealityStream to Cloud Run. Usage: ./deploy-cloud-run.sh [cpu|gpu]
# Needs: gcloud auth login; gcloud config set project <GOOGLE_PROJECT_ID>
set -euo pipefail
REGION="${GOOGLE_REGION:-us-central1}"
case "${1:-cpu}" in
  cpu) gcloud run deploy realitystream --source . --region "$REGION" --allow-unauthenticated \
         --memory 2Gi --timeout 900 ;;
  gpu) gcloud run deploy realitystream-gpu --source . --region "$REGION" --allow-unauthenticated \
         --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
         --cpu 8 --memory 32Gi --no-cpu-throttling --max-instances 1 --timeout 900 \
         --set-env-vars ENABLE_GPU=true ;;
  *) echo "usage: $0 [cpu|gpu]"; exit 1 ;;
esac
