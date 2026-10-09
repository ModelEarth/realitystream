# RealityStream plan

Linked from [ModelEarth/projects #63](https://github.com/ModelEarth/projects/issues/63) and [#77](https://github.com/ModelEarth/projects/issues/77).

## Status

| Step | State |
|---|---|
| run_models.py fixes: SMOTE shape, `state: all` (PR #60) | merged 2026-10-05 |
| cuML fallback when RAPIDS is absent (PR #61, also in colab cell 2) | merged 2026-10-05 |
| Fresh colab export trimmed into `run_models.py`; old `Run-Models-bkup` files removed | this PR |
| Data Commons key moved to `DATACOMMONS_API_KEY` (rotation still needed) | this PR |
| Cloud Run CPU service from this repo | next, needs access to the ModelEarth Google project |
| CloudRoot: Worker proxy route `/api/realitystream/run` | after the CPU service exists |
| Cloud Run GPU service, timing vs CPU | after CPU |
| Tree canopy generalization (#63), Data Commons two-column targets | Oct 22 and Nov 5 PRs |

Known colab issues found while trimming (2026-10-06): the data-loading cell forces the eye-blinks parameters regardless of the selected YAML, five cells call a venv path on one developer's Mac, and the SMOTE training cell imports cudf unconditionally. None of these are in `run_models.py`.

## Cloud architecture

Written 2026-10-05 against cloudroot `worker/README.md`, `cloud/run/config.yaml` and the Cloud Run GPU docs.

## Short answer

CloudRoot's Cloudflare Worker can serve the RealityStream front end, but it cannot run the models. The ML step belongs in a Cloud Run service built from this repo with Google Cloud buildpacks (no Dockerfile). The Worker proxies one API route to it.

```
browser --> cloud.model.earth --+-- /realitystream/*        static files from the realitystream submodule
                                |                           (index.html, parameters/, output/)
                                +-- /api/realitystream/run  Worker proxies to Cloud Run
                                                             |
                                                             v
                                           Cloud Run service "realitystream" (Flask models/main.py, run_models.py)
                                           CPU by default, optional NVIDIA L4 for cuML
```

## Why the Worker cannot host the Python backend

- CloudRoot deploys one Worker. Static files come from `worker/dist`, assembled from the webroot submodules by `scripts/build-static.mjs`. Only `/api/*` and `/sanity/*` reach Worker code, and that code is JavaScript (`worker/src/`).
- Cloudflare's Python Workers run on Pyodide. Pure-Python packages work, but scikit-learn, XGBoost, imbalanced-learn and cuML are compiled, and there is no GPU. Even the CPU path would not import.
- The free Workers plan allows 10 ms of CPU per request (the README calls this out for password hashing). A model run takes seconds to minutes.

So the Worker's job is to serve the pages and forward one request.

## What realitystream needs as a submodule

1. **Static content only in the build.** `build-static.mjs` already skips dot-files and tooling. Nothing in realitystream should break a static copy: `index.html`, `parameters/`, `output/` and the docs are fine as they are.
2. **One proxy route.** Add `worker/src/realitystream.js` that forwards `POST /api/realitystream/run` to the Cloud Run URL held in a Worker var `REALITYSTREAM_RUN_URL`, streaming the response back. About 20 lines, modeled on `src/sanity.js`.
3. **Front end calls its own origin.** Pages post YAML to `/api/realitystream/run`, so no CORS entry is needed, matching how the keys widget and chat already work.

## Cloud Run service from this repo

The repo contains `models/main.py` (Flask, `POST /run` takes a YAML body and returns JSON results, `GET /parameters`, `GET /health`) and a `Procfile` that starts it with gunicorn. Without a Dockerfile, `--source .` builds with Google Cloud buildpacks (Python from `.python-version`). `cloud/run/config.yaml` already names a project (`modelearth-run-models-1`, `us-central1`) and a service. A CPU deploy is one command:

```bash
gcloud run deploy realitystream --source . --region us-central1 --allow-unauthenticated --cpu 4 --memory 2Gi --timeout 3600 --max-instances 1
```

### GPU variant

Cloud Run supports NVIDIA L4 GPUs (driver 580, CUDA 13). Requirements from the docs:

| Requirement | Value |
|---|---|
| Minimum resources | 4 vCPU and 16 GiB (8 vCPU and 32 GiB recommended) |
| Billing mode | instance-based (`--no-cpu-throttling`); scale-to-zero still works |
| Default quota | 3 L4 GPUs per project per region on first deploy, zonal redundancy off |
| L4 regions | us-central1, us-east4, us-west1, europe-west1, europe-west4, asia-southeast1, asia-south1 and others |
| Price | $0.0001867 per GPU-second without zonal redundancy, about $0.67 per hour, plus CPU and memory |

Deploy:

```bash
gcloud run deploy realitystream-gpu --source . --region us-central1 \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling --max-instances 1 --timeout 900 \
  --set-env-vars ENABLE_GPU=true
```

The image needs RAPIDS wheels for the GPU path (`cuml-cu12`, `cudf-cu12`, `cupy-cuda12x`). Keep them in a separate `requirements-gpu.txt` so the CPU image stays small. `run_models.py` already falls back to scikit-learn when cuML is absent (PR #61 logic), so one codebase serves both services.

### Cost expectation

A 50-state XGBoost run is seconds of GPU time. At per-second billing with scale-to-zero, a few hundred runs a month costs single-digit dollars. Cold start on a GPU instance is the main latency (tens of seconds while the container and CUDA libraries load), so the front end should show progress rather than block.

### Testing without the ModelEarth project

Any Google Cloud account with billing enabled can run the commands above; new accounts get $300 credit. Put the project id in the shared env file as `GOOGLE_PROJECT_ID` (already a key in `automation/.env.example`) and nothing in the repo changes.

## Keys and settings

- Backend settings come from the env file named by `automation/paths.yaml`, as `cloud/run/utils/env_paths.py` and the notebook already do. On Cloud Run the same names arrive as service env vars or Secret Manager references.
- When a key is missing, `models/main.py` returns a JSON error with a `how_to_get_it` field (for example a link to the Data Commons API key page), and the page renders that text instead of a stack trace.
- The current notebook export has a Google Data Commons API key hardcoded in the Data Commons cells. It should move to `DATACOMMONS_API_KEY` in the env file and be rotated.

## Proposed order

1. CPU service deployed from this repo and reachable at a Cloud Run URL (one command, no code change).
2. Worker proxy route in cloudroot plus `REALITYSTREAM_RUN_URL` var.
3. Front end button on `/realitystream/` that posts the selected parameters YAML and shows the JSON results.
4. GPU service, `requirements-gpu.txt`, timing comparison against the CPU service.
