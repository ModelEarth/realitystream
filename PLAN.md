# RealityStream plan

Linked from [ModelEarth/projects #63](https://github.com/ModelEarth/projects/issues/63) and [#77](https://github.com/ModelEarth/projects/issues/77).

## Status

| Step | State |
|---|---|
| run_models.py fixes: SMOTE shape, `state: all` (PR #60) | merged 2026-10-05 |
| cuML fallback when RAPIDS is absent (PR #61, also in colab cell 2) | merged 2026-10-05 |
| Fresh colab export trimmed into `run_models.py`; old `Run-Models-bkup` files removed (PR #62) | merged 2026-10-08 |
| Data Commons key moved to `DATACOMMONS_API_KEY` | done; rotation of the old key still needed |
| Cloud Run CPU service (`models/main.py`, buildpacks, 4 vCPU, Firestore usage totals, Team Passphrase) | live 2026-10-08 at realitystream-839376296196.us-central1.run.app |
| Run panel on model.earth/realitystream/models calling the service | live 2026-10-09 |
| Colab backup kept as a `.py` export (`models/run-models-colab.py`), no `.ipynb` in the repo | this PR |
| CloudRoot: Worker proxy route `/api/realitystream/run` | next |
| Cloud Run GPU service, timing vs CPU | after the proxy |
| Tree canopy generalization (#63): canopy and all-years YAMLs run unchanged, join aliases, FIPS padding, NAICS names | merged 2026-10-17 (PR #68, Oct 22 item) |
| `timeseries` section in the canopy YAML | file is missing in tree-canopy; decision needed on what it should do |
| Data Commons two-column targets | Nov 5 PR |
| Repo layout: move what's in use from the cloud repo, and files out of the root (below) | done 2026-10-21; Cloud Run revision realitystream-00013-2jz |
| Model parameter editing, from the Streamlit forms (below) | planned 2026-10-21, after the layout move |
| Archive the streamlit repo | done 2026-10-21; removed from webroot's `.gitmodules` |

How the raw export compares with `run_models.py` (2026-10-09): the export is 16,845 lines (10,612 code lines, 168 cells) and defines `train_multiple_models` seven times with 16 cuML import blocks; the module is 795 lines (646 code lines, 29 functions and classes), about 6 percent of the export, with the same report output.

Known colab issues found while trimming (2026-10-06): the data-loading cell forces the eye-blinks parameters regardless of the selected YAML, five cells call a venv path on one developer's Mac, and the SMOTE training cell imports cudf unconditionally. None of these are in `run_models.py`.

## Repo layout

Proposed 2026-10-21. Everything the RealityStream deploy and pages use moves into this repo, and the root keeps only the landing page, docs and the files that must sit there.

### In use from the cloud repo

| From `cloud/run/` | Used by | To |
|---|---|---|
| `js/param-input.js` | `realitystream/index.html`, `models/index.html`, `models/home.html`, `profile/item/index.html` | `js/param-input.js` |
| `css/param.css` | the same four pages | `css/param.css` |
| `README.md`, section `#googleaccountautomation` (Google account setup, Team Passphrase, running more models) | `MORE_RUNS_URL` in `models/main.py`, `js/run-panel.js`, AGENTS.md | `deploy/README.md`, shown by `deploy/index.html` |
| `README.md`, section "Local key store" | README.md "Keys" | `deploy/README.md` |
| `utils/keystore.py`, `routes/keys_routes.py` (the localhost:8100/keys paste page) | Stores keys that `run_models.get_env` reads from the OS credential store | `api/keystore.py`, with `/keys` added to `api/main.py` only when it isn't running on Cloud Run (`K_SERVICE` unset) |

Everything else in `cloud/run/` (`app.py`, the notebook-executor, Papermill, `Dockerfile`, `deploy.sh`, `config.yaml`, `page.html`) is the older process and stays behind.

### Proposed layout

```
realitystream/
  index.html about.md projects.md        Stay: the landing page loads both .md files by relative path
  README.md AGENTS.md PLAN.md CLAUDE.md LICENSE
  Procfile requirements.txt .python-version .gcloudignore
                                          Stay: buildpacks read them from the deploy root; CI reads the last two
  parameters/                             Stays: raw GitHub URLs in parameter-paths.csv and the Colab point here
  api/        main.py, home.html           From models/
              run_models.py                From the root
              keystore.py                  From cloud/run/utils/
              model-params.yaml            New (model parameter editing)
  deploy/     deploy-cloud-run.sh, add-team-members.sh, read-setting.sh   From the root
              README.md, index.html        From cloud/run/README.md
  js/         model-select.js, run-panel.js
              param-input.js               From cloud/run/js/
  css/        param.css                    From cloud/run/css/
  archive/    the five root .ipynb files, admin/run_models.py renamed run_models_archive.py (older Colab export)
  models/     index.html, README.md, run-models-colab.py, location-forest/, random-bits-forest/
  tests/  input/  output/  process/  shap/  cuML/   Unchanged
```

`api/` puts the whole Cloud Run service in one folder, separate from the `models/` page. `.gcloudignore` then lists what to upload (`api/`, `parameters/`, the RBF binary) instead of excluding `models/*` with exceptions.

### Path changes the move needs

- `js/param-input.js`: `RS_WEBROOT` becomes `new URL('../../', ...)` (the file is one folder shallower).
- `realitystream/index.html` loads `js/param-input.js` directly, which drops the `externalSite` workaround for when the cloud repo is missing. `models/index.html` and `api/home.html` use `../js/` and `../css/`, or `{{ASSET_BASE}}realitystream/js/`.
- `Procfile`: `api.main:app`. `api/main.py`: put its own folder on `sys.path`, and update `HOME_PAGE_FILE`, `SOURCE_URL` and `MORE_RUNS_URL` (to `realitystream/deploy/#googleaccountautomation`).
- `api/run_models.py`: the RBF binary path and `env_file_path()` are relative to `HERE`, so they become relative to the realitystream folder.
- `deploy/*.sh`: `cd` to the realitystream folder before `gcloud run deploy --source .`, and update the RBF path.
- `tests/test_run_models.py`: `sys.path` points to `../api`.
- README.md, about.md and AGENTS.md: update the CLI commands to `python api/run_models.py ...` and the file paths.

### Order

1. **realitystream**: move the files and apply the path changes. Run `python -m pytest tests` and `python api/main.py`, check http://localhost:8887/realitystream/ and http://localhost:8887/realitystream/models/, then push.
2. **Cloud Run**: run `./deploy/deploy-cloud-run.sh`, because `home.html` and the `Procfile` moved. Check `/`, `/health` and a one-model run.
3. **profile**: point `item/index.html` at `/realitystream/js/param-input.js` and `/realitystream/css/param.css`.
4. **cloud**: delete the moved files, and replace the README section with a link to `realitystream/deploy/`. Mark the rest of `cloud/run` as the older process.
5. **webroot**: update `start cloud` in AGENTS.md to point here (`python api/main.py`, `deploy/deploy-cloud-run.sh`). The streamlit repo is archived and removed from `.gitmodules`.

## Model parameter editing

Written 2026-10-21. Let users set model hyperparameters and the train/test split on [Run Models](models/), building on the forms in the streamlit repo (`utils/model_page.py`).

### What the Streamlit forms have

| Section | Controls | Carry over? |
|---|---|---|
| Vectorizer | n-grams, max features, Count / TF-IDF | No. Text-only (job listings); our features are tabular. Revisit if a text dataset is added. |
| Data | test size 10-100%, oversampling yes/no | Yes: test size (capped at 50%), `random_state`. Oversampling already exists as the SMOTE choice in `run-panel.js`. |
| Logistic Regression | penalty, solver, random state, n_jobs | Yes, with valid combinations only. Drop `n_jobs`. |
| Random Forest | criterion, n_estimators, max_features, max_depth, random state, n_jobs | Yes. Drop `n_jobs`, fix the criterion list. |
| SVM | kernel, random state | Yes, plus `C`. |
| Random Bits Forest, XGBoost, MLP | none (RBF crashes: the name doesn't match and it returns `none`) | Add: RBF `number_of_trees`; XGBoost and MLP fixed values or search ranges. |
| Predict page, Explore page | job text prediction, dataset charts | No. A data preview (rows, class balance, missing values) could come later. |

### Problems to avoid

- The Streamlit app reruns the whole script on every widget change and calls `run_notebook_from_github` outside the button, so changing a dropdown starts a notebook run. Here, a run starts only from the Run button.
- Invalid options: RF criterion `elasticnet`, LR `l1` or `elasticnet` with solvers that reject them, test size 100%.
- `n_jobs` sliders up to 1000. That's a server setting (4 vCPU on Cloud Run), not a model choice.
- Values exist only in widget state, so a run can't be shared or repeated. Here they go into the YAML, which already travels in the hash, the Copy link button and the report folder (`parameters.yaml`).
- Streamlit ran notebooks with Papermill (the older `cloud` process). We keep `api/run_models.py` behind `api/main.py`. Don't deploy Streamlit as a second service.

### Approach

The front end stays independent of the API. The form edits the YAML in `#paramText pre`, the same way `model-select.js` edits `models:`, and `POST /run` stays unchanged. Only `run_models.py` learns to read new YAML keys.

```yaml
split:
  test_size: 0.2        # 0.1 to 0.5
  random_state: 42
search:
  n_iter: 20            # RandomizedSearchCV tries for xgboost and mlp
hyperparameters:
  rfc: {n_estimators: 100, max_depth: 8, criterion: gini, max_features: sqrt}
  lr: {penalty: l2, solver: lbfgs, C: 1.0}
  svm: {kernel: rbf, C: 1.0}
  xgboost: {learning_rate: 0.1}   # A fixed value drops that key from the search
  mlp: {hidden_layer_sizes: [100], activation: relu}
  rbf: {number_of_trees: 200}
```

Leaving out a section keeps the current behavior, as the `evaluation:` block already does.

One schema file, `api/model-params.yaml`, lists each model's fields: type, default, min and max, options, a one-line help text, which options need which others (LR penalty by solver), and whether cuML accepts the field. The form is built from it, and the API validates against it, so the page and the server can't disagree.

### Steps

**API** (deploy before the front end; until then the old service ignores these keys without saying so)

1. `run_models.py`: pass `hyperparameters[key]` into `make_model`, `split` into `split_data`, and `search.n_iter` into `train_models`. Remove fixed keys from `_param_grid`. Add `params_used` per model to the summary, using `best_params_` when a search ran.
2. `run_models.validate_parameters(params, schema)`: reject unknown fields, out-of-range values and invalid combinations, with a message naming the field. `api/main.py` returns it as HTTP 400 (it is a 500 today). Without the Team Passphrase, a YAML containing `split`, `search` or `hyperparameters` gets HTTP 403 `key_needed`, as more than one model does now.
3. `GET /model-params` in `api/main.py` returns the schema, so the page shows the field list the deployed service validates against. A 404 tells the page an older service is running, and it hides the panel.
4. `.gcloudignore`: nothing to add, since `api/` is uploaded whole.
5. Tests in `tests/test_run_models.py`: overrides reach the estimator, fixed keys leave the grid, validation errors, and an unchanged YAML gives unchanged results.
6. `./deploy/deploy-cloud-run.sh`, then check `GET /model-params` on the live URL.

**Front end** (push to GitHub; no redeploy except for `home.html`)

7. New `js/model-params.js` renders a "Model Settings" panel: a Split row, then one card for each selected model, following `model-select.js` changes. It writes the YAML sections, reads them back when the YAML is edited by hand, and only writes values that differ from the defaults, so the YAML stays short.
8. Controls: number fields with steppers and range hints rather than sliders, selects that grey out invalid options (with the reason), "Default" beside each changed value, and a "Search" or "Fixed" switch for XGBoost and MLP fields. Help text comes from the schema. Without the Team Passphrase, every field still shows with its default, but inactive, under the note "Enter the Team Passphrase to change model settings." The fields become active when `run-panel.js` accepts the passphrase, so it needs to announce key changes (for example a `rsKeyChange` event). Lay out by panel width, include `.dark` styles, and use the Panel Menu Toggle System from `localsite/js/localsite.js`.
9. `js/param-input.js`: add `split`, `search` and `hyperparameters` to `RS_YAML_ROOTS`, so shared links keep them and the Reset button counts them.
10. `models/index.html`: add the panel between Models and Settings. In `api/home.html`, do the same and redeploy, since Cloud Run serves that page.
11. `js/run-panel.js`: show `params_used` in the results table, and show 400 messages beside the field they name.

**Later**

- Data preview before a run: rows, class balance, missing values (the Streamlit Explore page, for our datasets).
- A runtime estimate from `n_estimators × n_iter × rows`, compared with the minutes left today.
- Streamlit's Predict page needs a saved model. Consider it once reports store fitted models.

### Decisions (2026-10-21)

- Without the Team Passphrase, users see every field, inactive, with its default. Runs without the passphrase use the defaults.
- The YAML section is named `hyperparameters`.
- Archive the streamlit repo (ModelEarth/streamlit). Done 2026-10-21, and the `streamlit` submodule is removed from webroot.

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

- CloudRoot deploys one Worker. Static files come from `worker/dist`, assembled from the webroot submodules by `scripts/build-static.mjs`. Only `/api/*` and `/ingest/*` reach Worker code, and that code is JavaScript (`worker/src/`).
- Cloudflare's Python Workers run on Pyodide. Pure-Python packages work, but scikit-learn, XGBoost, imbalanced-learn and cuML are compiled, and there is no GPU. Even the CPU path would not import.
- The free Workers plan allows 10 ms of CPU per request (the README calls this out for password hashing). A model run takes seconds to minutes.

So the Worker's job is to serve the pages and forward one request.

## What realitystream needs as a submodule

1. **Static content only in the build.** `build-static.mjs` already skips dot-files and tooling. Nothing in realitystream should break a static copy: `index.html`, `parameters/`, `output/` and the docs are fine as they are.
2. **One proxy route.** Add `worker/src/realitystream.js` that forwards `POST /api/realitystream/run` to the Cloud Run URL held in a Worker var `REALITYSTREAM_RUN_URL`, streaming the response back. About 20 lines: `fetch(new Request(target, request))` with the target built from that var.
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
