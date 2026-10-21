# RealityStream: AI CLI Guide

Active processes for the Google Cloud deployment. Plans and status: [PLAN.md](PLAN.md). Older UI notes: [CLAUDE.md](CLAUDE.md). Webroot rules (secrets, git, styling) still apply: [../AGENTS.md](../AGENTS.md).

## Layout

| Folder | Contents |
|---|---|
| root | `index.html` (loads `about.md`, `projects.md`), docs, and `Procfile`, `requirements.txt`, `.python-version`, `.gcloudignore`, which buildpacks read from the deploy root |
| `api/` | The Cloud Run service: `main.py` (Flask), `run_models.py` (pipeline and CLI), `home.html` (the service's home page), `keystore.py` (local key store, command line) |
| `deploy/` | `deploy-cloud-run.sh`, `add-team-members.sh`, `read-setting.sh`, and `README.md` with the Google Cloud setup steps |
| `js/`, `css/` | Shared front end: `param-input.js`, `model-select.js`, `run-panel.js`, `model-params.js`, `param.css` |
| `models/` | Run Models page (`index.html`), the model list (`README.md`), model subpages, `run-models-colab.py` |
| `parameters/` | Preset YAMLs. Stays put: raw GitHub URLs in `parameter-paths.csv` and the Colab point here |
| `archive/` | Older notebooks and `run_models_archive.py`. Not used by the service |

## Two independent parts

| Part | Files | Ships by |
|---|---|---|
| **Front end** (static) | `index.html`, `models/index.html`, `js/*`, `css/*`, `models/README.md` | Pushing to GitHub (model.earth Pages). Local: http://localhost:8887/realitystream/models/ |
| **API** (Cloud Run service `realitystream`) | `api/*`, `parameters/*.yaml`, `requirements.txt`, `Procfile`, `.python-version` | `./deploy/deploy-cloud-run.sh` |

The front end only sends YAML (the text of `#paramText pre`) to `POST /run`. Anything a page edits must end up in that YAML, and `api/run_models.py` must read it.

`api/home.html` loads the shared JS and CSS from `{{ASSET_BASE}}realitystream/` (https://model.earth/ by default). JS changes reach the Cloud Run home page once they are pushed; a redeploy doesn't carry them.

## API (`api/main.py`)

`GET /` (HTML for browsers, usage JSON otherwise), `GET /health`, `GET /key`, `GET /github-token` (whether GITHUB_REPORTS_TOKEN, or header X-GitHub-Token, can push, cached 5 minutes), `GET /parameters`, `GET /model-params`, and `POST /run[?upload=1][&smote=0|1]`.

- Team Passphrase: header `X-API-Key` matching `REALITYSTREAM_API_KEY` (Secret Manager `realitystream-api-key`). With it: 20 runs a day, any models, uploads. Without it: 1 run a day of 1 model. The page's Active mode also limits a run to 2 models (`MODEL_LIMIT` in `js/run-panel.js`); the service doesn't.
- Model settings: the YAML's `split`, `search` and `hyperparameters` sections, defined in `api/model-params.yaml` (served at `/model-params`). `run_models.validate_parameters` checks them, and `/run` returns HTTP 400 with `field` before the run is counted. Without the Team Passphrase, any of these sections gets HTTP 403 `key_needed`. A schema default for a field that isn't searched must match `make_model()`; a test checks this.
- Reports: each run writes a `{year}/run-{date}T{time}` folder with an `index.html` (localsite's `start/template/report.html`) and a `files.json` listing the folder's files, which the page shows (localsite's `displayCharts()`). The local API saves it in the webroot's `reports/` (the reports repo; `REPORTS_DIR` overrides) and returns `output_path`, which the page links after the Run button. On Cloud Run the folder is temporary. With the Team Passphrase and `GITHUB_REPORTS_TOKEN` set, the folder is also pushed to GitHub modelearth/reports, and `github.message` (shown below the Run button) says whether it was. A token saved in the browser on the webroot's keys page (`keys/#need=GITHUB_REPORTS_TOKEN`, provider `github-reports`) is sent by the page as header `X-GitHub-Token` on `/run` and `/github-token`, and is pushed with instead of the service's, with or without the Team Passphrase. `GET /` reports `github_token` (true or false, never the value). `models: []` trains no models and still writes the folder.
- Daily cost cap `DAILY_COST_LIMIT_USD` (default $0.20). Usage totals in Firestore (collection `usage`), reset at midnight Eastern.
- CORS allows only https://model.earth and http://localhost:8887 (and 127.0.0.1:8887). Add origins in `ALLOWED_ORIGINS`.
- Live URL: https://realitystream-kwr4qrkopq-uc.a.run.app (also https://realitystream-839376296196.us-central1.run.app), set as `RS_RUN_API` in `js/run-panel.js`.

## Deploy

Setup steps (project creation, billing, auth, Team Passphrase secret, local key store) are in [deploy/README.md](deploy/README.md), shown at model.earth/realitystream/deploy/.

```bash
cd realitystream
./deploy/deploy-cloud-run.sh        # CPU service "realitystream": 4 vCPU, 2 GiB, timeout 3600, max 1 instance
./deploy/deploy-cloud-run.sh gpu    # "realitystream-gpu": NVIDIA L4 (not yet in use, see PLAN.md)
```

- Needs `gcloud auth login`. Settings come from the environment or the env file named in `automation/paths.yaml`, through `deploy/read-setting.sh`: `GOOGLE_PROJECT_ID` (required), `GOOGLE_REGION` (default us-central1), `GOOGLE_TEAM_EMAILS`, `GOOGLE_TEAM_ROLES`. Never print their values.
- The script also runs `deploy/add-team-members.sh` (IAM roles for team emails), downloads the Random Bits Forest binary into `models/random-bits-forest/rbf/`, copies `GITHUB_REPORTS_TOKEN` from the env file to the `github-reports-token` secret when it is missing or differs, and attaches the `realitystream-api-key` and `github-reports-token` secrets when they exist.
- No Dockerfile: buildpacks read `requirements.txt`, `Procfile` and `.python-version`.
- `.gcloudignore` uploads only `api/`, `parameters/`, the buildpack files and the RBF binary. A new file the service reads elsewhere needs a `!` line there. Check with `gcloud meta list-files-for-upload .`
- `--max-instances 1` keeps the run and cost counters exact. Keep it.
- Other settings `api/run_models.py` reads: `GITHUB_REPORTS_TOKEN` (for `upload=1`), `DATACOMMONS_API_KEY` (YAMLs with `dcid`), `ENABLE_GPU`. On Cloud Run, pass them with `--set-env-vars` or `--set-secrets`.
- Logs: `gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=realitystream' --limit 50 --project "$GOOGLE_PROJECT_ID"`

## Local testing

```bash
cd realitystream
pip install -r requirements.txt
python api/run_models.py parameters/parameters-blinks.yaml   # pipeline only, writes report/
python api/main.py                                    # API on port 8080, usage counted in memory
ASSET_BASE=http://localhost:8887/ python api/main.py  # home page loads your local shared JS
python -m pytest tests                                # also runs in CI (.github/workflows/tests.yml)
```

On localhost, http://localhost:8887/realitystream/models/ uses the local API when one answers on port 8080 (`apiBase()` in `js/run-panel.js`), so runs save their reports in your webroot's `reports/` folder. Add `?api=https://realitystream-kwr4qrkopq-uc.a.run.app/` to use Cloud Run instead (`?api=` works only on localhost). The Common Settings panel (`js/model-params.js`) appears only when that API serves `/model-params`.

## Related repos

- **cloud**: the older Papermill notebook-executor (`cloud/run/app.py`, port 8100, `start cloud`). Not part of this deploy. Its `cloud/run/index.html` loads `realitystream/js/param-input.js` in the `cloud-run` mode.
- **profile**: `profile/item/index.html` also loads `js/param-input.js` and `css/param.css` from here.
