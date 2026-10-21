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

## Next

1. **Rotate the old Data Commons key.** It was hardcoded in the Colab export; the code now reads `DATACOMMONS_API_KEY`.
2. **Worker proxy in cloudroot** (below): `POST /api/realitystream/run` forwarded to Cloud Run.
3. **GPU service** `realitystream-gpu` with a `requirements-gpu.txt`, and timing against the CPU service.
4. **`timeseries` section** in the canopy YAML: the file is missing in tree-canopy. Decide what it should do.
5. **Data Commons two-column targets.**

The Colab export still has issues `run_models.py` doesn't: its data-loading cell forces the eye-blinks parameters, five cells call a venv path on one developer's Mac, and the SMOTE cell imports cudf unconditionally.

## Model Settings

The Common Parameters panel on [Run Models](models/) (`js/model-params.js`) edits `split` and `search`, and each model's `hyperparameters` are in its Details in the Models panel, shown by its "N Parameters" button (or "Parameters" beside Expand all for every model, `#expand=params`). Leaving a section out keeps the default behavior.

```yaml
split: {test_size: 0.2, random_state: 42}
search: {n_iter: 20}                    # RandomizedSearchCV tries for xgboost and mlp
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
