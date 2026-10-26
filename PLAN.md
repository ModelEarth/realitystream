# RealityStream plan

Linked from [ModelEarth/projects #63](https://github.com/ModelEarth/projects/issues/63) and [#77](https://github.com/ModelEarth/projects/issues/77). How the deploy works today: [AGENTS.md](AGENTS.md).

## Done

| Work | Done |
|---|---|
| `run_models.py` trimmed from the Colab export (about 6 percent of its 16,845 lines, same reports); SMOTE, `state: all` and cuML fallback fixes (PRs #60-62) | 2026-10-06 |
| Colab backup kept as `models/run-models-colab.py`, no `.ipynb` (PR #63) | 2026-10-09 |
| Evaluation fixes: stratified split, scaling, leak-free SMOTE, state-held-out cross-validation, CI tests (PRs #64-67) | 2026-10-10 |
| Tree canopy generalization (#63): canopy, all-years and bee YAMLs run unchanged (PR #68) | 2026-10-10 |
| Cloud Run CPU service `realitystream`, with Firestore usage totals and the Team Passphrase | 2026-10-08 |
| Run panel on model.earth/realitystream/models calling the service | 2026-10-09 |
| Repo layout: the service in `api/`, deploy scripts in `deploy/`, shared `js/` and `css/` moved in from the cloud repo, old notebooks in `archive/` | 2026-10-21 |
| Streamlit repo archived and removed from webroot's `.gitmodules` | 2026-10-21 |
| Model Settings: `split`, `search` and `hyperparameters` editing (below) | 2026-10-21 |
| Run reports saved locally in `reports/{year}/run-{date-time}/` with a link after Run, pushed to GitHub when `GITHUB_REPORTS_TOKEN` is set, runs with no models | 2026-10-21 |
| Preview / Active modes on Run Models, a 2-model limit in Active mode, each model's hyperparameters in its Details | 2026-10-21 |
| Report charts (below): PNGs from `api/charts.py` for all 6 models, failed models and data warnings in each report's README.md, binary-safe push, `models: all` and comma lists read; deployed and checked on Cloud Run | 2026-10-26 |
| Run IDs (below): `/run?stream=1` sends the run ID at once and keep-alive lines; `GET /run/<run_id>` for a dropped connection | 2026-10-26 |
| Feature importance options: RBF's chart as a checkbox in its Parameters, and `importance` checkboxes (prescreen, time budget) for SVM and MLP charts on more than 200 features | 2026-10-26 |
| `metrics_comparison.png` (six metrics, 3 x 2); report folders named after the preset (`2026/eye-blinks-all-six-models-{date-time}`); a Rerun Report button on each report opens Run Models with its parameters.yaml and a link back | 2026-10-26 |
| `feature_importance_{model}.csv` for every charted model (rbf when checked), industry features named industry first ("Mining, Quarrying, Oil and Gas Extraction - Establishments") with a NAICS column | 2026-10-26 |

## Report charts

Reports from 2026-10-21 held only README.md, model-options.csv and parameters.yaml. Why, and what changed:

- **No charts were written.** The Colab's plotting functions aren't in its export, so the trimmed `run_models.py` wrote only CSVs. `api/charts.py` now writes `metrics_dashboard.png`, `metrics_comparison.png` (a 3 x 2 grid, one metric a panel, like the 2025 Eye Blinks dashboard), `roc_curves_comparison.png`, `confusion_matrices.png`, `training_time_comparison.png` and `feature_importance_{model}.png`, which the report page (localsite's `displayCharts()`) shows from `files.json`. Feature importance is native for rfc, xgboost and lr, and by permutation for svm and mlp. Over 200 features is skipped unless the `importance` section's checkboxes say otherwise: `prescreen` shuffles the 200 that track the target most closely (ANOVA F on the training rows), and `time_budget` shuffles all of them when the first 64 project a finish within `budget_seconds`, falling back to prescreen when both are checked. Uncapped, bees MN (10,260 features) took 69 s for MLP and 145 s for SVM locally, likely 5 and 11 minutes on Cloud Run. Locally on Eye Blinks that took 2.5 s for SVM, 0.05 s for MLP and under 0.1 s for the others. RBF's chart is optional (`hyperparameters: rbf: feature_importance: true`, a checkbox in its Parameters): its binary trains again on every prediction, so its shuffled copies of the test rows are stacked into one call (up to 100,000 rows), which trains it once more, about 2.5 more minutes for Eye Blinks on Cloud Run.
- **A failed model left no trace.** Those runs chose only `rbf` on the default bees preset: Maine's 16 counties have one top-20-percent county, so training saw one class, and RBF needs two (as do LR and SVM). The Cloud Run logs confirm it: "rbf failed: RandomBitsForest supports binary targets only" on 12 training rows. On a Mac RBF always fails, since its binary runs on Linux only. The report's README.md now gives the rows, the target classes and each failure's reason, and the `/run` reply's `errors` shows on the page.
- **Pushes would have broken images.** `upload_reports` sent every file as UTF-8 text. Binary files now go up as base64 blobs.
- **Other fixes found while testing:** SMOTE inside the search's cross-validation was sized for the whole training split and failed on small classes (bees MN), RBF trained twice per pass (predict, then predict_proba), and `models: rbf,rfc,xgboost` and `models: all` in four presets were read as unknown models.

Checked 2026-10-26 with Python 3.9 / scikit-learn 1.6 and Python 3.14 / scikit-learn 1.9 (47 tests):

| YAML | Rows, features | Models charted | Time |
|---|---|---|---|
| parameters-blinks.yaml with all models | 14,980, 14 | lr, rfc, svm, mlp, xgboost | 40 s |
| parameters-all-years-bees-MN.yaml with all models | 84, 10,260 | lr, rfc, svm, mlp, xgboost | 30 s |
| parameters-datacenter-locations.yaml with all models | 7, 8 | lr, rfc, svm, mlp, xgboost | 4 s |
| parameters.yaml, bees in Maine only (before) | 16, 57 | rfc, mlp, xgboost (one class in training) | 2 s |
| parameters.yaml, bees in CT, ME, MA, NH, RI, VT (now), all models | 59, 57 (11 positive) | lr, rfc, svm, mlp, xgboost | 6 s |

Data problems (too few rows, one class, a feature file that didn't load) are listed under "Data warnings" in the report's README.md and in the `/run` reply's `warnings`. A run whose data can't be loaded still writes a README.md naming the error.

On Cloud Run (revision realitystream-00023, deployed 2026-10-26), pushed to modelearth/reports:

| Run | Result | Report |
|---|---|---|
| Eye Blinks, rbf only | accuracy 0.944, ROC-AUC 0.989; 148 s a pass (315 s before RBF stopped training twice) | [run-2026-10-26T05-01-35](https://model.earth/reports/2026/run-2026-10-26T05-01-35/) |
| Bees in 6 New England states, all 6 models | every model charted, RBF included; 47 s | [run-2026-10-26T05-07-12](https://model.earth/reports/2026/run-2026-10-26T05-07-12/) |

The PNGs reach GitHub intact. Locally, a test runs RBF's code path through the charts with a stand-in binary.

## Run IDs for long runs

Eye Blinks with all six models ([run-2026-10-26T05-54-58](https://model.earth/reports/2026/run-2026-10-26T05-54-58/)) finished on Cloud Run in 17.6 minutes (MLP's search 5 minutes a pass, RBF 2.5) and pushed its report, but the caller never got the reply: nothing was sent until the end, and the idle connection was dropped along the way.

Now `POST /run?stream=1`, which the page sends, answers at once with the run ID and the report's name, sends a line every 15 seconds, then the result. The run stays inside the open request because Cloud Run bills by request: CPU is allocated only while one is open, so a run left in the background after an immediate reply would barely move. If the connection drops anyway, the page asks `GET /run/<run_id>` until the result is in. Each ask waits up to 25 seconds for it, so a request stays open and the run keeps its CPU. Tested locally 2026-10-26: the ID came at once, keep-alive lines every 15 seconds, and a run whose connection was closed after the first line was still collected by ID.

## Next

1. **Try a long run from the page**, such as Eye Blinks (All Six Models), now that the run IDs are deployed (revision realitystream-00025). With RBF's feature_importance checked, check the time it adds: one more training, plus predicting 71,000 stacked rows for Eye Blinks.
2. **SVC `probability=True`** is deprecated in scikit-learn 1.9 and removed in 1.11; switch to `CalibratedClassifierCV(SVC(), ensemble=False)`.
3. **Rotate the old Data Commons key.** It was hardcoded in the Colab export; the code now reads `DATACOMMONS_API_KEY`.
4. **Worker proxy in cloudroot** (below): `POST /api/realitystream/run` forwarded to Cloud Run.
5. **GPU service** `realitystream-gpu` with a `requirements-gpu.txt`, and timing against the CPU service.
6. **`timeseries` section** in the canopy YAML: the file is missing in tree-canopy. Decide what it should do.
7. **Data Commons two-column targets.**

The Colab export still has issues `run_models.py` doesn't: its data-loading cell forces the eye-blinks parameters, five cells call a venv path on one developer's Mac, and the SMOTE cell imports cudf unconditionally.

## Model Settings

The Common Parameters panel on [Run Models](models/) (`js/model-params.js`) edits `split`, `search` and `importance`, and each model's `hyperparameters` are in its Details in the Models panel, shown by its "N Parameters" button (or "Parameters" beside Expand all for every model, `#expand=params`). Leaving a section out keeps the default behavior.

```yaml
split: {test_size: 0.2, random_state: 42}
search: {n_iter: 20}                    # RandomizedSearchCV tries for xgboost and mlp
importance: {prescreen: true, time_budget: true, budget_seconds: 60}  # Permutation charts for over 200 features
hyperparameters:
  rfc: {n_estimators: 100, max_depth: 8}
  xgboost: {learning_rate: 0.1}         # A fixed value leaves the search
```

- `api/model-params.yaml` defines every field. The API validates against it and serves it at `/model-params`, and the panel is built from it, so the two can't disagree.
- The panel writes only values that differ from the defaults, mirrors them in the URL hash for shared links, and reads hand edits of the YAML back.
- Preview mode (the default, `js/run-panel.js`) lets anyone check any models and edit every field, with Run Models gray. Active mode shows the Team Passphrase field and greys out the rest until it's entered, then allows 2 models a run (`MODEL_LIMIT`; the page enforces it, not the service).
- `/run` answers 403 to a YAML with these sections without the Team Passphrase. An invalid value gets 400 naming the field, which the page marks, opening that model's Details.
- Results show each model's `params_used`.
- Logistic Regression keeps a `penalty` choice. scikit-learn 1.8 deprecated that argument, so `run_models._lr_kwargs` translates it to `l1_ratio` and `C`.

Later:

- A data preview before a run: rows, class balance, missing values.
- A runtime estimate from `n_estimators × n_iter × rows`, compared with the minutes left today.
- A Predict page, once reports store fitted models.

## Cloud architecture

CloudRoot's Cloudflare Worker can serve the pages but can't run the models. Python Workers run on Pyodide, without compiled scikit-learn, XGBoost or cuML. The free plan also allows 10 ms of CPU per request. So models run on Cloud Run, and the Worker forwards one route:

```
browser --> cloud.model.earth --+-- /realitystream/*        static files from the realitystream submodule
                                +-- /api/realitystream/run  Worker proxies to Cloud Run (api/main.py)
```

The proxy is about 20 lines, `worker/src/realitystream.js`. It forwards to a Worker var `REALITYSTREAM_RUN_URL` and streams the response back. Pages then post to their own origin, with no CORS entry needed.

### GPU service

`./deploy/deploy-cloud-run.sh gpu` deploys `realitystream-gpu` on an NVIDIA L4. `run_models.py` falls back to scikit-learn without cuML, so one codebase serves both services. Before using it:

- Add `requirements-gpu.txt` (`cuml-cu12`, `cudf-cu12`, `cupy-cuda12x`), keeping the CPU image small.
- Requirements: at least 4 vCPU and 16 GiB, instance-based billing (`--no-cpu-throttling`), and a default quota of 3 L4s per region.
- Cost: about $0.67 an hour per GPU, billed by the second with scale-to-zero. A few hundred short runs a month cost single-digit dollars.
- Cold starts take tens of seconds, so the page should show progress.
