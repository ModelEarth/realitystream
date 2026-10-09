#!/usr/bin/env bash
# Deploy RealityStream to Cloud Run. Usage: ./deploy-cloud-run.sh [cpu|gpu]
# Needs: gcloud auth login, and GOOGLE_PROJECT_ID in the environment or in the env file named in automation/paths.yaml
# No Dockerfile: Google Cloud buildpacks read requirements.txt, Procfile and .python-version.
# --max-instances 1 keeps main.py's per-day run and spending counters exact.
set -euo pipefail
cd "$(dirname "$0")"

source ./read-setting.sh

PROJECT="$(read_setting GOOGLE_PROJECT_ID)"
REGION="$(read_setting GOOGLE_REGION)"
REGION="${REGION:-us-central1}"
if [ -z "$PROJECT" ]; then echo "Set GOOGLE_PROJECT_ID in the shared env file"; exit 1; fi
./add-team-members.sh

# Bundle the Random Bits Forest binary (Linux x86-64) in the upload so runs don't download it.
# run_models.RandomBitsForest looks for it at this path; it's gitignored and kept by .gcloudignore.
RBF="models/random-bits-forest/rbf/rbf"
if [ ! -x "$RBF" ]; then
  echo "Downloading the Random Bits Forest binary for the build"
  tmp="$(mktemp -d)"
  curl -sSL --fail -A realitystream -o "$tmp/rbf.zip" \
    "${RBF_BINARY_URL:-https://downloads.sourceforge.net/project/random-bits-forest/rbf.zip}"
  mkdir -p "$(dirname "$RBF")"
  unzip -q -o "$tmp/rbf.zip" rbf -d "$(dirname "$RBF")"
  chmod 755 "$RBF"
  rm -rf "$tmp"
fi

case "${1:-cpu}" in
  cpu) gcloud run deploy realitystream --source . --project "$PROJECT" --region "$REGION" --allow-unauthenticated \
         --memory 2Gi --timeout 900 --max-instances 1 ;;
  # GPU instances bill per instance (about $0.0004/s for L4 + 8 vCPU + 32 GiB), so the cap uses that rate
  gpu) gcloud run deploy realitystream-gpu --source . --project "$PROJECT" --region "$REGION" --allow-unauthenticated \
         --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
         --cpu 8 --memory 32Gi --no-cpu-throttling --max-instances 1 --timeout 900 \
         --set-env-vars ENABLE_GPU=true,COST_PER_SECOND_USD=0.0004 ;;
  *) echo "usage: $0 [cpu|gpu]"; exit 1 ;;
esac
