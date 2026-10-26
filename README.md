Our [Run Models CoLab](input/industries) provides Logistic Regression, Support Vector Machines (SVM), MLP, XGBoost and two Random Forests.

Paths for **features** and **targets** are joined based on location IDs (state or county FIPS, etc.) or brain regions for [eye blink predictions](models/random-bits-forest/).

For Features, the UI picks `features.dcid` if it exists otherwise `features.data` otherwise the filename from `features.path`.  
For Targets, it picks `targets.dcid` if it exists otherwise `targets.data` otherwise the filename from `targets.path`.

Our main input is currently industry features by county ID (FIPS) for exploring environmental impact targets like [bee data](../bee-data) and [tree canopy](https://model.earth/tree-canopy/).  
We are also preping data for [International Trade Flow](https://model.earth/profile/trade).


[run-models-colab.py](models/run-models-colab.py) is the .py export of the [Run Models CoLab](https://colab.research.google.com/drive/1EFGEwh6v1-oYFTcNVCBOyl_HtfEC6INd?usp=sharing), kept for reference. The trimmed, runnable version of that notebook is [api/run_models.py](api/run_models.py) (see "Running RealityStream Locally" below).

- [Run Models CoLab](input/industries)
- [Random Forests for Healthy Bees](../bee-data/)
- [Random Bits Forest for Eye Blinks](models/random-bits-forest/)

The Run Models colab merges feature and target datasets in Pandas.  
Training files can be reviewed by setting save_training=True.

Select one or more [models](models/), then paste the resulting yaml in the Run Models CoLab.

Learn about our [cuML GPU speed enhancements - and SMOTE balancing of our classes](cuML)
We're using [SHAP to explain our model predictions](shap)

<h2>Design your Stream</h2>

**Bee YAML Updated** - Changed [bee data](/bee-data) target to bees-targets-top-20-percent.csv in parameters yaml. This new "colony density" target uses the top 20% of counties with the highest bee population density (rather than top colony growth between years, as was used in bees-targets.csv).

<!--
Density file: bees-targets-top-20-percent.csv. Shashank worked from bees-population-usda.csv
(We previously used growth over time with the file bees-targets.csv)
-->

In the CoLab, a select menu allows you to choose default parameter yaml paths set in [parameter-paths.csv](https://github.com/ModelEarth/realitystream/blob/main/parameters/parameter-paths.csv).
[parameters-simple.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-simple.yaml) - Bee density for 2020 for Maine and New York by top industries (2-digits)
[parameters.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters.yaml) - Predicts bee density by industry (6-digit NAICS) for New England states 2017 to 2021
[parameters-gdc.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-gdc.yaml) - Google Data Commons - Predicting states using demographics
[parameters-zip.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-zip.yaml) - Needs zip code target. Uses bee population density.  
[parameters-blinks.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-blinks.yaml) - Uses only features dataset (which contains the target column).

<!--

[parameters-years.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-years.yaml) - For testing with multiple years and states (currently same as parameters.yaml).

TO DO: Web page displaying US counties at risk of increased poverty - Use Google Data Commons API for FIPS county poverty target data and international target data. Pull with an [Observable Data Loader](../../../timelines/observable/)
-->

TO DO: Top ten counties in each state likely to have [declining tree canopy](/data-pipeline/research/canopy/)

TO DO: Within Run Models, add python pull from Google Data Commons API for population, education levels, income/poverty levels to use as both features and targets. - IN PROGRESS

TO DO: Use [Tensorflow.org](https://www.tensorflow.org/js/demos) for [Neural Network predictions](https://www.tensorflow.org/s/results/?q=neural%20networks) with our training data.

<!--
### Javascript Display in Tabulator

In javascript, we'll populate "Density" for each county and append it as a column in Tabulator. [Tabulator work in progress](/data-pipeline/timelines/tabulator/).

Density = Population / Km2

Density can also be thought of as PopPerKm2 (divided by 1000)
100,000 people living in an 80 Km2 county = 1250 people per Km2 = Density of 1.25
When displaying, we will multiply Density and Population by 1000.

Append 0 or 1 to the "y" column. Prior y column in community forecasting: y=1 when the current year’s poverty had no decline from the prior year AND the next year’s poverty increased by 2% or more.


Applied in
prep/all/zcta_2016.SQL.txt

-- Change from prior year is steady (0%) or increasing, change to next year is increasing by 2% or more.

CASE
      WHEN (prior1.poverty - p.poverty) >= 0 AND (p.poverty - next.poverty) >= 2 THEN 1
      ELSE 0
END

AS y -- the povertyBinary for >= 2% in coming year, and no decline for current year.
-->
## Running RealityStream Locally (CLI)

`api/run_models.py` is the colab pipeline as one importable module. It fetches features and targets from a parameters.yaml, trains the requested models twice (plain and SMOTE-balanced), and writes the same `report/` folder the colab pushes to [modelearth/reports](https://github.com/modelearth/reports).

    pip install -r requirements.txt
    python api/run_models.py parameters/parameters.yaml
    python api/run_models.py parameters/parameters-blinks.yaml --report-dir report --n-iter 20
    python api/run_models.py parameters/parameters.yaml --upload      # also push report/ to modelearth/reports

Model keys for the YAML `models` list: `lr`, `rfc`, `rbf`, `svm`, `mlp`, `xgboost`. `rbf` is the Random Bits Forest binary, which runs on Linux only (Colab, Docker, Cloud Run); on Windows and macOS it is skipped with a warning.

`report/` contains `model_performance_report_no_smote.csv`, `model_performance_report_smote.csv`, `feature_importance_xgboost.csv`, `parameters.yaml`, `model-options.csv`, `README.md` and `index.html`, plus PNG charts from `api/charts.py`: `metrics_dashboard.png` (six metrics grouped by metric), `metrics_comparison.png` (the same six, one panel each with values), `roc_curves_comparison.png`, `confusion_matrices.png`, `training_time_comparison.png` and `feature_importance_{model}.png` (rfc, xgboost and lr natively; svm and mlp by permutation, up to 200 features unless `importance` sets `prescreen` or `time_budget`; rbf by permutation when its `feature_importance` hyperparameter is true, which trains it once more). The charts show the pass that trained more models (with SMOTE on a tie). `README.md` gives the rows, target classes and the reason any model wasn't trained, such as RBF on a Mac (its binary runs on Linux only) or a training split with a single class.

### Cross-validation (optional)

Add an `evaluation:` block to the parameters YAML to also score with k-fold cross-validation. Omit it and the run is unchanged.

    evaluation:
      cv: stratified   # or: group
      folds: 5

`stratified` uses `StratifiedKFold(shuffle=True)` over all rows. `group` uses `GroupKFold` with each county's **state** (FIPS // 1000) as the group, so whole states are held out — this exposes geographic leakage (state-held-out scores sit below stratified); with fewer groups than folds it falls back to `stratified`. Scaling and SMOTE are fit inside each fold (fixed model settings, no nested search, to stay within the Cloud Run cost cap). Results — fold mean ± std for ROC-AUC, PR-AUC, balanced accuracy and macro F1, plus baseline lift — go to `model_performance_cv.csv` and the `"cv"` key of the run summary; the existing report files are untouched.

### Model settings (optional)

Add `split`, `search`, `importance` or `hyperparameters` sections to change the test share, the seed, the search size, feature importance for wide data or a model's settings. Leave them out and the run is unchanged. [api/model-params.yaml](api/model-params.yaml) lists every setting with its default and range, and the run stops with a message naming the setting when a value isn't allowed.

    split:
      test_size: 0.3        # 0.1 to 0.5 (default 0.2)
      random_state: 7       # seeds the split, models, SMOTE and search (default 42)
    search:
      n_iter: 10            # RandomizedSearchCV tries for xgboost and mlp (default 20)
    importance:             # svm and mlp permutation charts on more than 200 features (skipped by default)
      prescreen: true       # shuffle only the 200 features that track the target most closely
      time_budget: true     # shuffle every feature when that fits in budget_seconds (tried first)
      budget_seconds: 60
    hyperparameters:
      rfc: {n_estimators: 200, max_depth: 12}
      lr: {penalty: l1, solver: liblinear, C: 0.5}
      xgboost: {learning_rate: 0.05}   # a fixed value leaves the search; the rest are still searched

Each model in the run summary lists its `params_used`, including the values the search chose. On the Cloud Run API, changing these settings needs the Team Passphrase.

### Keys

Settings are read one name at a time: first from the environment, then from the OS credential store (see [api/keystore.md](api/keystore.md)), then from the env file named by `automation/paths.yaml` in the [webroot](https://model.earth/webroot). The whole file is never loaded.

- `GITHUB_REPORTS_TOKEN` for `--upload` (GitHub token with write access to the reports repo)
- `GITHUB_REPORTS_REPO` (default `modelearth/reports`) and `GITHUB_REPORTS_SITE` (default `https://model.earth/reports/`), to push somewhere else, for example your own fork while testing
- `DATACOMMONS_API_KEY` for parameter files that use `dcid` and for `process/canopy/canopy-process.py` (request one at [apikeys.datacommons.org](https://apikeys.datacommons.org/))
- `ENABLE_GPU=true` to use cuML models when RAPIDS is installed (falls back to scikit-learn otherwise)

When a key is missing, the CLI and the Flask `/run` endpoint explain where to get it.

#### Getting a GitHub token for report pushes

1. Open [github.com/settings/tokens](https://github.com/settings/tokens) and choose **Generate new token (classic)**. Name it `realitystream reports`, set an expiry, tick the **repo** scope, and generate it. (A fine-grained token also works: give it **Contents: Read and write** on the reports repo.)
2. Store it: `python api/keystore.py set GITHUB_REPORTS_TOKEN`, or add `GITHUB_REPORTS_TOKEN=<token>` to your env file. For the Run Models web page instead, save it on the [keys page](https://model.earth/keys/#need=GITHUB_REPORTS_TOKEN); the browser sends it as `X-GitHub-Token` and it never reaches a file.
3. Check it: `python -c "import sys; sys.path.insert(0,'api'); from run_models import check_reports_token; print(check_reports_token())"` asks GitHub whether the token can write to the reports repo, without printing the token.

#### Testing a push against your own fork

Fork [modelearth/reports](https://github.com/ModelEarth/reports), then:

    GITHUB_REPORTS_REPO=<you>/reports GITHUB_REPORTS_SITE=https://<you>.github.io/reports/       python api/run_models.py parameters/parameters.yaml --upload

The run prints the folder it pushed, `<year>/run-<timestamp>/`, under your fork. Nothing reaches modelearth/reports until you drop the two variables.

### Cloud Run

`api/main.py` wraps the pipeline as `POST /run` (body: parameters.yaml text or JSON `{"parameters": "parameters-blinks.yaml"}`, `?upload=1` to push the report), plus `GET /parameters` and `GET /health`. `deploy/deploy-cloud-run.sh cpu|gpu` deploys it without a Dockerfile: Google Cloud buildpacks read `requirements.txt`, `Procfile` and `.python-version`, and `.gcloudignore` limits the upload to `api/` and `parameters/`. Setup steps: [deploy/](deploy/). See also [AGENTS.md](AGENTS.md) and [PLAN.md](PLAN.md).

### Tests

    python -m pytest tests
