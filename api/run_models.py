"""
RealityStream: run the "Run Models" pipeline from a parameters.yaml file.

This is the trimmed, importable version of the Run Models colab
(models/run-models-colab.py is its raw export). It produces the same report folder the colab
pushes to github.com/modelearth/reports:

    report/
      README.md (what ran, and why any model wasn't trained), index.html, files.json, parameters.yaml, model-options.csv
      model_performance_report_no_smote.csv
      model_performance_report_smote.csv
      feature_importance_xgboost.csv
      metrics_dashboard.png, roc_curves_comparison.png, confusion_matrices.png,
      training_time_comparison.png, feature_importance_{model}.png (api/charts.py)

Usage:
    python api/run_models.py parameters/parameters.yaml
    python api/run_models.py parameters/parameters-blinks.yaml --upload
    python api/run_models.py https://raw.githubusercontent.com/.../parameters.yaml

Settings (GITHUB_REPORTS_TOKEN, DATACOMMONS_API_KEY, ENABLE_GPU) are read one
name at a time from the environment, then from the env file named by
automation/paths.yaml in the webroot. The whole env file is never loaded.
"""

import argparse
import base64
import contextvars
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import OrderedDict
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# The realitystream folder: this file resides in its api/ folder
REALITYSTREAM_DIR = os.path.dirname(HERE)
RANDOM_STATE = 42
REPORTS_REPO = "modelearth/reports"  # Default for GITHUB_REPORTS_REPO
REPORT_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/ModelEarth/localsite/refs/heads/main/start/template/report.html"
)
NAICS6_NAMES_URL = "https://github.com/ModelEarth/concordance/raw/master/data-raw/6-digit_2017_Codes.xlsx"
NAICS2_NAMES_URL = "https://raw.githubusercontent.com/ModelEarth/community-data/master/us/id_lists/naics2.csv"
RBF_BINARY_URL = "https://downloads.sourceforge.net/project/random-bits-forest/rbf.zip"

# ---------------------------------------------------------------------------
# Settings: one value by name, never the whole env file
# ---------------------------------------------------------------------------

KEY_HELP = {
    "GITHUB_REPORTS_TOKEN": (
        "GitHub personal access token with write access to modelearth/reports. "
        "Create one at https://github.com/settings/tokens (classic token, 'repo' scope), "
        "then add GITHUB_REPORTS_TOKEN=<token> to the env file named in automation/paths.yaml."
    ),
    "DATACOMMONS_API_KEY": (
        "Google Data Commons API key. Request one at https://docs.datacommons.org/api/ "
        "(see 'Get an API key'), then add DATACOMMONS_API_KEY=<key> to the env file named in "
        "automation/paths.yaml."
    ),
}


class MissingKey(Exception):
    """A required setting is absent. `how_to_get_it` is safe to show to end users."""

    def __init__(self, name):
        self.name = name
        self.how_to_get_it = KEY_HELP.get(name, f"Add {name}=<value> to the env file named in automation/paths.yaml.")
        super().__init__(f"{name} is not set. {self.how_to_get_it}")


def env_file_path():
    """Absolute path of the shared env file, from automation/paths.yaml, or None."""
    for automation in (os.path.join(REALITYSTREAM_DIR, "..", "automation"), os.path.join(REALITYSTREAM_DIR, "..", "..", "automation")):
        paths_yaml = os.path.join(automation, "paths.yaml")
        if not os.path.exists(paths_yaml):
            continue
        with open(paths_yaml, encoding="utf-8") as fh:
            match = re.search(r"^\s*env_file:\s*(.+)$", fh.read(), re.MULTILINE)
        if match:
            value = re.sub(r"\s+#.*$", "", match.group(1)).strip().strip("\"'")
            return os.path.abspath(os.path.join(automation, value))
    return None


def get_env(name, default=None):
    """Return one setting: os.environ, then the OS credential store, then the single matching line of the env file."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # local OS credential store, filled by api/keystore.py or the local /keys page (optional dependency)
        import keyring
        value = keyring.get_password("modelearth", name)
        if value:
            return value
    except Exception:
        pass
    path = env_file_path()
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("\"'") or default
    return default


def reports_repo():
    """The GitHub repo runs are pushed to: GITHUB_REPORTS_REPO (owner/name), else modelearth/reports."""
    return get_env("GITHUB_REPORTS_REPO") or REPORTS_REPO


def reports_site(repo=None):
    """The web address the reports repo is served at, ending in /: GITHUB_REPORTS_SITE, else model.earth
    for modelearth, else the owner's GitHub Pages ({owner}.github.io/{name}/)."""
    repo = repo or reports_repo()
    site = get_env("GITHUB_REPORTS_SITE")
    if site:
        return site.rstrip("/") + "/"
    owner, name = repo.split("/", 1)
    if owner.lower() == "modelearth":
        return f"https://model.earth/{name}/"
    if name.lower() == f"{owner.lower()}.github.io":
        return f"https://{name.lower()}/"
    return f"https://{owner.lower()}.github.io/{name}/"


_RUN_WARNINGS = contextvars.ContextVar("run_warnings", default=None)  # Per thread, as gunicorn runs 4


def warn(message):
    """Print a data warning, and keep it for the run's README.md and /run reply (run_pipeline collects them)."""
    print(f"  [WARN] {message}")
    collected = _RUN_WARNINGS.get()
    if collected is not None and message not in collected:
        collected.append(message)


def require_env(name):
    value = get_env(name)
    if not value:
        raise MissingKey(name)
    return value


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

_GPU = None


def gpu_enabled():
    """True when a GPU runtime is requested AND cuML imports. Cached."""
    global _GPU
    if _GPU is None:
        wanted = "COLAB_GPU" in os.environ or str(get_env("ENABLE_GPU", "")).lower() in ("1", "true") \
            or str(get_env("GOOGLE_CLOUD_GPU_SERVICE", "")).lower() in ("1", "true")
        if wanted:
            try:
                import cuml  # noqa: F401
            except ImportError:
                print("[GPU] cuML is not installed; running CPU (scikit-learn) models.")
                wanted = False
        _GPU = wanted
    return _GPU


def to_cpu(data):
    """cupy / cudf -> numpy / pandas; everything else unchanged."""
    mod = type(data).__module__
    if mod.startswith("cupy"):
        return data.get()
    if mod.startswith("cudf"):
        return data.to_pandas()
    return data


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MODEL_KEYS = {
    "lr": "lr", "logisticregression": "lr",
    "rfc": "rfc", "randomforest": "rfc",
    "rbf": "rbf", "randombitsforest": "rbf",
    "svm": "svm", "mlp": "mlp",
    "xgboost": "xgboost", "xgb": "xgboost",
}
MODEL_TITLES = {
    "lr": "Logistic Regression", "rfc": "Random Forest Classifier", "rbf": "Random Bits Forest",
    "svm": "Support Vector Machine", "mlp": "Multi-Layer Perceptron", "xgboost": "XGBoost",
}


class DictToObject:
    """Recursively convert a dict to an object with dot-notation access."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, DictToObject(v) if isinstance(v, dict) else v)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, DictToObject) else v for k, v in vars(self).items()}


# ---------------------------------------------------------------------------
# Model settings: the YAML's split, search and hyperparameters sections
# ---------------------------------------------------------------------------

MODEL_PARAMS_PATH = os.path.join(HERE, "model-params.yaml")
SETTINGS_SECTIONS = ("split", "search", "hyperparameters")
_MODEL_PARAMS = None


def model_params_schema():
    """Fields a parameters.yaml may set in split, search and hyperparameters (api/model-params.yaml)."""
    global _MODEL_PARAMS
    if _MODEL_PARAMS is None:
        with open(MODEL_PARAMS_PATH, encoding="utf-8") as fh:
            _MODEL_PARAMS = yaml.safe_load(fh)
    return _MODEL_PARAMS


class ParameterError(ValueError):
    """A split, search or hyperparameters value that model-params.yaml doesn't allow; field names it."""

    def __init__(self, field, message):
        super().__init__(f"{field}: {message}")
        self.field = field


def _check_value(field, spec, value):
    """One YAML value checked against its model-params.yaml spec; returns it normalized."""
    kind = spec["type"]
    if kind == "choice":
        if value is None and "none" in spec["options"]:
            value = "none"
        if str(value) not in [str(o) for o in spec["options"]]:
            raise ParameterError(field, f"{value} isn't an option; choose {', '.join(map(str, spec['options']))}")
        return str(value)
    if kind == "layers":
        sizes = [value] if isinstance(value, (int, str)) and not isinstance(value, bool) else value
        if isinstance(sizes, (list, tuple)):  # A URL hash gives "50,50" as text
            sizes = [int(s) if isinstance(s, str) and s.strip().isdigit() else s for s in sizes]
        if not isinstance(sizes, (list, tuple)) or not 1 <= len(sizes) <= 3:
            raise ParameterError(field, "give 1 to 3 layer sizes, for example [100] or [50, 50]")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int) or not spec["min"] <= size <= spec["max"]:
                raise ParameterError(field, f"each layer size must be a whole number from {spec['min']} to {spec['max']}")
        return list(sizes)
    if isinstance(value, bool):
        raise ParameterError(field, f"must be a number, not {value}")
    if kind == "int":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ParameterError(field, f"must be a whole number, not {value}")
    else:
        try:
            value = float(value)  # also accepts text such as 1e-4, which YAML reads as a string
        except (TypeError, ValueError):
            raise ParameterError(field, f"must be a number, not {value}") from None
    if not spec["min"] <= value <= spec["max"]:
        raise ParameterError(field, f"must be from {spec['min']} to {spec['max']}, not {value}")
    return value


def _check_fields(prefix, fields, values):
    """Check one section's {field: value} mapping, including the only_with and valid_with rules."""
    if not isinstance(values, dict):
        raise ParameterError(prefix, "must be a list of setting: value lines")
    checked = {}
    for name, value in values.items():
        if name not in fields:
            raise ParameterError(f"{prefix}.{name}", f"unknown setting; choose from {', '.join(fields)}")
        checked[name] = _check_value(f"{prefix}.{name}", fields[name], value)

    def effective(name):  # The YAML's value, else the default
        return checked.get(name, fields[name].get("default"))

    for name in checked:
        for other, allowed in (fields[name].get("only_with") or {}).items():
            if effective(other) not in allowed:
                raise ParameterError(f"{prefix}.{name}", f"applies only when {other} is {' or '.join(allowed)} "
                                                         f"({other} is {effective(other)})")
    for name, spec in fields.items():
        for other, table in (spec.get("valid_with") or {}).items():
            allowed = table.get(effective(other))
            if (name in checked or other in checked) and allowed is not None and effective(name) not in allowed:
                raise ParameterError(f"{prefix}.{name}", f"{effective(name)} doesn't work with {other} {effective(other)}; "
                                                         f"choose {' or '.join(allowed)}, or another {other}")
    return checked


def validate_parameters(params):
    """Check the YAML's split, search and hyperparameters sections against model-params.yaml.

    Returns {"split": {...}, "search": {...}, "hyperparameters": {model_key: {...}}} holding only the values
    the YAML sets, normalized (model aliases become keys). Raises ParameterError naming the field.
    Hyperparameters for models the YAML doesn't run are checked too, then ignored by training.
    """
    schema = model_params_schema()
    settings = {"split": {}, "search": {}, "hyperparameters": {}}
    for section in ("split", "search"):
        if params.get(section) is not None:
            settings[section] = _check_fields(section, schema[section], params[section])
    hyper = params.get("hyperparameters")
    if hyper is None:
        return settings
    if not isinstance(hyper, dict):
        raise ParameterError("hyperparameters", "must list models, each with setting: value lines")
    for model, values in hyper.items():
        key = MODEL_KEYS.get(str(model).lower())
        if key is None:
            raise ParameterError(f"hyperparameters.{model}", f"unknown model; choose from {', '.join(schema['hyperparameters'])}")
        if values is not None:
            settings["hyperparameters"][key] = _check_fields(f"hyperparameters.{model}", schema["hyperparameters"][key], values)
    return settings


def _sklearn_at_least(major, minor):
    import sklearn

    return tuple(int(n) for n in re.findall(r"\d+", sklearn.__version__)[:2]) >= (major, minor)


def _lr_kwargs(values, cuml=False):
    """Logistic Regression arguments for the installed library. The YAML names a penalty (l2, l1, elasticnet,
    none); scikit-learn 1.8 deprecated `penalty` (removed in 1.10) in favor of l1_ratio, with C=inf for none."""
    values = dict(values)
    penalty = values.pop("penalty", None)
    if penalty == "elasticnet":
        values.setdefault("l1_ratio", model_params_schema()["hyperparameters"]["lr"]["l1_ratio"]["default"])
    if penalty is None:
        return values
    if cuml:
        values["penalty"] = penalty
    elif not _sklearn_at_least(1, 8):
        values["penalty"] = None if penalty == "none" else penalty
    elif penalty == "none":
        values.update(C=np.inf, l1_ratio=0.0)
    elif penalty in ("l1", "l2"):
        values["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    return values


def apply_hyperparameters(model, key, values, gpu=False):
    """Set the YAML's hyperparameters (validated) on a make_model() estimator and return it.
    On the GPU path, fields cuML doesn't accept are left out with a warning."""
    if not values:
        return model
    fields = model_params_schema()["hyperparameters"][key]
    values = dict(values)
    if gpu:
        dropped = [name for name in values if fields[name].get("gpu") is False]
        for name in dropped:
            values.pop(name)
        if dropped:
            print(f"  [WARN] cuML {key} ignores {', '.join(dropped)}")
    if key == "lr":
        values = _lr_kwargs(values, cuml=gpu)
    if "hidden_layer_sizes" in values:
        values["hidden_layer_sizes"] = tuple(values["hidden_layer_sizes"])
    return model.set_params(**values)


def _json_value(value):
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None  # Browsers can't parse Infinity in JSON
    return value


def params_used(key, model):
    """The model's values for its model-params.yaml fields, JSON-safe. Searched fields show what the
    search chose; a library default left as None shows the schema default."""
    fields = model_params_schema()["hyperparameters"].get(key, {})
    try:
        current = model.get_params()
    except Exception:
        return {}
    used = {name: _json_value(current[name]) if current[name] is not None else spec.get("default")
            for name, spec in fields.items() if name in current and "only_with" not in spec}
    if key == "lr":
        penalty = current.get("penalty")
        if penalty == "deprecated":  # scikit-learn 1.8+: l1_ratio and C carry the penalty
            ratio = current.get("l1_ratio") or 0
            penalty = ("none" if current.get("C") == np.inf else "l2" if ratio == 0
                       else "l1" if ratio == 1 else "elasticnet")
        elif penalty is None:
            penalty = "none"
        used["penalty"] = penalty
        if penalty == "elasticnet":
            used["l1_ratio"] = _json_value(current.get("l1_ratio"))
        if penalty == "none":
            used.pop("C", None)
    if key == "svm" and used.get("kernel") == "poly":
        used["degree"] = _json_value(current.get("degree"))
    return used


PARAMETER_PATHS_URL = "https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameter-paths.csv"


def default_parameters_url():
    """First entry of parameter-paths.csv, the colab's default selection."""
    for name, link in csv.reader(StringIO(requests.get(PARAMETER_PATHS_URL, timeout=60).text)):
        return link
    raise ValueError("parameter-paths.csv is empty")


def load_parameters(yaml_path_or_url):
    """Load the YAML (local path or URL); normalise `models` to canonical lowercase keys.

    A YAML without a `models` line runs rfc. An empty one (`models: []`, as the Run Models page writes
    when every model is unchecked) trains none, and the run still writes its report folder.
    `models` may also be text: one key, keys separated by commas, or `all`.
    """
    if yaml_path_or_url.startswith(("http://", "https://")):
        text = requests.get(yaml_path_or_url, timeout=60).text
    else:
        with open(yaml_path_or_url, encoding="utf-8") as fh:
            text = fh.read()
    params = yaml.safe_load(text) or {}
    models = params.get("models", ["rfc"]) or []
    if isinstance(models, str):  # "rbf,rfc,xgboost" in several presets; "all" for every model
        models = list(MODEL_TITLES) if models.strip().lower() == "all" else [m.strip() for m in models.split(",")]
    keys = []
    for m in models:
        key = MODEL_KEYS.get(str(m).lower())
        if key is None:
            print(f"[WARN] Unknown model '{m}', choose from {sorted(set(MODEL_KEYS.values()))}")
        elif key not in keys:
            keys.append(key)
    params["models"] = keys
    return params


def _common_column(param):
    """Join column: features/targets `common` or `id`, then `join.key`, then top-level `common`, else Fips."""
    for holder in ("features", "targets"):
        obj = getattr(param, holder, None)
        for attr in ("common", "id"):
            if obj is not None and getattr(obj, attr, None):
                return getattr(obj, attr)
    join = getattr(param, "join", None)
    if join is not None and getattr(join, "key", None):
        return join.key
    return getattr(param, "common", None) or "Fips"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def build_feature_urls(param):
    """Expand the features.path template over naics x year x state."""
    template = getattr(param.features, "path", None)
    if not template:
        return []
    if "{" not in template:
        return [template]
    naics = getattr(param.features, "naics", []) or [0]
    if not isinstance(naics, list):
        naics = [naics]
    start, end = getattr(param.features, "startyear", None), getattr(param.features, "endyear", None)
    years = list(range(start, end + 1)) if start and end else [0]
    raw = getattr(param.features, "state", "")
    if isinstance(raw, list):
        states = raw
    elif str(raw).strip().lower() == "all":
        states = US_STATES
    elif raw:
        states = [s.strip() for s in str(raw).split(",")]
    else:
        states = [""]
    urls = []
    for state in states:
        for year in years:
            for n in naics:
                try:
                    urls.append(template.format(naics=n, year=year, state=state))
                except KeyError:
                    pass
    return urls


def fetch_csv(url):
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def load_gdc_data(param):
    """Google Data Commons pull when features/targets carry `dcid`. Returns (features_df, targets_df)."""
    has_f = hasattr(getattr(param, "features", None), "dcid")
    has_t = hasattr(getattr(param, "targets", None), "dcid")
    if not (has_f or has_t):
        return None, None
    try:
        from datacommons_client import DataCommonsClient
    except ImportError as exc:
        raise ImportError("pip install datacommons-client to use dcid parameters") from exc
    client = DataCommonsClient(api_key=require_env("DATACOMMONS_API_KEY"))

    def pull(section):
        dcids = section.dcid if isinstance(section.dcid, list) else [section.dcid]
        variables = getattr(section, "variables", ["Count_Person"])
        variables = variables if isinstance(variables, list) else [variables]
        year = getattr(section, "year", "LATEST")
        return client.observations_dataframe(
            variable_dcids=variables, date=str(year), entity_dcids=dcids
        )

    features_df = targets_df = None
    if has_f:
        obs = pull(param.features)
        if obs is not None and not obs.empty:
            obs["entity"] = obs["entity"].astype(str).str.replace("geoId/", "", regex=False)
            features_df = obs.pivot_table(index="entity", columns="variable", values="value", aggfunc="median")
            features_df.index.name = _common_column(param)
            features_df = features_df.reset_index()
            print(f"  [OK] GDC features: {features_df.shape}")
    if has_t:
        obs = pull(param.targets)
        if obs is not None and not obs.empty:
            col = getattr(param.targets, "common", "Fips")
            obs[col] = (obs["entity"].astype(str)
                        .str.replace("zip/", "", regex=False)
                        .str.replace("geoId/", "", regex=False)
                        .str.replace("postalCode/", "", regex=False))
            agg = obs.groupby(col)["value"].sum().reset_index().rename(columns={"value": "Target"})
            agg["Target"] = (agg["Target"] > 0).astype(int)
            targets_df = agg[[col, "Target"]]
            print(f"  [OK] GDC targets: {targets_df.shape}")
    return features_df, targets_df


def load_data(param, return_groups=False):
    """Fetch and merge features + targets. Returns (X, y) with numeric X, or (X, y, groups) when
    return_groups is True. groups is the state FIPS (county join key // 1000) for group CV, or None
    when it can't be derived (e.g. an inline target column with no join key)."""
    features_df, target_df = load_gdc_data(param)

    if features_df is None:
        urls = build_feature_urls(param)
        if not urls:
            raise ValueError("No feature URLs could be constructed from parameters.")
        frames = []
        for url in urls:
            try:
                frames.append(fetch_csv(url))
                print(f"  [OK] Loaded features: {url}")
            except Exception as exc:
                warn(f"Features not loaded from {url}: {exc}")
        if not frames:
            raise FileNotFoundError("Could not load any feature files.")
        features_df = pd.concat(frames, ignore_index=True)

    inline_target = getattr(param.features, "target_column", None)
    target_path = getattr(getattr(param, "targets", None), "path", None)

    if target_df is None and (inline_target or not target_path):
        col = inline_target if inline_target in features_df.columns else "y"
        if col not in features_df.columns:
            raise ValueError(f"Target column '{inline_target}' not in features and no targets.path given.")
        X_inline, y_inline = _numeric(features_df.drop(columns=[col])), features_df[col]
        return (X_inline, y_inline, None) if return_groups else (X_inline, y_inline)

    if target_df is None:
        target_df = fetch_csv(target_path)
        print(f"  [OK] Loaded targets: {target_path}")

    wanted = getattr(param.targets, "column", None) or getattr(param.targets, "target_column", None)
    candidates = ([wanted] if wanted else []) + ["Target", "target", "y"]
    target_col = next((c for c in candidates if c in target_df.columns), None)
    if target_col is None:
        raise ValueError(f"Cannot find target column ({'/'.join(candidates)}) in targets data.")
    if wanted and target_col != wanted:
        warn(f"targets.column '{wanted}' isn't in the targets file, so '{target_col}' is used.")

    common = _common_column(param)
    f_cols = {c.lower(): c for c in features_df.columns}
    t_cols = {c.lower(): c for c in target_df.columns}
    f_key, t_key = f_cols.get(common.lower()), t_cols.get(common.lower())
    if f_key is None or t_key is None:
        raise ValueError(f"Common column '{common}' must exist in both features and targets.")
    features_df[f_key] = _id_text(features_df[f_key], common)
    target_df[t_key] = _id_text(target_df[t_key], common)

    merged = features_df.merge(target_df[[t_key, target_col]], left_on=f_key, right_on=t_key, how="inner")
    if merged.empty:
        raise ValueError("Merge produced 0 rows. Check the common column values.")
    drop = {f_key, t_key, target_col}
    X_merged, y_merged = _numeric(merged.drop(columns=[c for c in drop if c in merged.columns])), merged[target_col]
    if return_groups:
        # The join key was cast to str above; convert back to integer FIPS, then // 1000 for the state.
        groups = (pd.to_numeric(merged[f_key], errors="coerce") // 1000).astype("Int64")
        return X_merged, y_merged, groups
    return X_merged, y_merged


def _id_text(series, name):
    """Join keys as text; county FIPS zero-padded to 5 so 1001 and 01001 match."""
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(5) if name.lower() == "fips" else text


def _numeric(X):
    X = X.loc[:, [c for c in X.columns if not str(c).startswith("Unnamed:")]]  # index column from to_csv()
    dropped = X.select_dtypes(exclude=["number"]).columns.tolist()
    if dropped:
        print(f"  [WARN] Dropping non-numeric columns: {dropped}")
    return X.select_dtypes(include=["number"])


def apply_smote(X_train, y_train):
    """SMOTE oversampling. Keeps every column; caps k_neighbors for tiny minorities."""
    from imblearn.over_sampling import SMOTE

    counts = pd.Series(y_train).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        print("  [WARN] SMOTE needs two classes with at least 2 samples each; skipping.")
        return None, None
    X_imp = X_train.fillna(X_train.mean()).fillna(0)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, int(counts.min()) - 1))
    return sm.fit_resample(X_imp, y_train)


# ---------------------------------------------------------------------------
# Random Bits Forest (external binary, Linux only)
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402


class RandomBitsForest(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper for the RBF binary (https://sourceforge.net/projects/random-bits-forest/)."""

    def __init__(self, number_of_trees=200, bin_path=None):
        self.number_of_trees = number_of_trees
        self.bin_path = bin_path

    def fit(self, X, y):
        if platform.system() != "Linux":
            raise RuntimeError("The Random Bits Forest binary runs on Linux only (Colab, Docker, Cloud Run).")
        self._le = LabelEncoder()
        self._y = self._le.fit_transform(np.asarray(to_cpu(y)).ravel()).astype(float)
        if len(self._le.classes_) != 2:
            raise ValueError("RandomBitsForest supports binary targets only.")
        self._X = np.asarray(to_cpu(X), dtype=float)
        self.n_features_in_ = self._X.shape[1]
        self._last_proba = None
        return self

    def predict_proba(self, X):
        X = np.asarray(to_cpu(X), dtype=float)
        # Each call trains the forest again, so predict() after predict_proba() on the same rows reuses the result
        rows = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
        cached = getattr(self, "_last_proba", None)
        if cached is not None and cached[0] == rows:
            return cached[1]
        binary = self._ensure_binary()
        work = tempfile.mkdtemp(prefix="rbf_")
        try:
            paths = {k: os.path.join(work, f"{k}.csv") for k in ("trainx", "trainy", "testx", "testYhat")}
            pd.DataFrame(self._X).to_csv(paths["trainx"], header=False, index=False)
            pd.DataFrame(self._y).to_csv(paths["trainy"], header=False, index=False)
            pd.DataFrame(X).to_csv(paths["testx"], header=False, index=False)
            cmd = [binary, "-n", str(self.number_of_trees), paths["trainx"], paths["trainy"], paths["testx"], paths["testYhat"]]
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"RBF failed ({proc.returncode}): {proc.stderr[:500]}")
            p1 = np.clip(pd.read_csv(paths["testYhat"], header=None).iloc[:, 0].to_numpy(float), 0, 1)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self._last_proba = (rows, np.column_stack([1 - p1, p1]))
        return self._last_proba[1]

    def predict(self, X):
        return self._le.inverse_transform((self.predict_proba(X)[:, 1] >= 0.5).astype(int))

    def _ensure_binary(self):
        path = self.bin_path or os.path.join(REALITYSTREAM_DIR, "models", "random-bits-forest", "rbf", "rbf")
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
        target = os.path.dirname(path)
        os.makedirs(target, exist_ok=True)
        url = os.environ.get("RBF_BINARY_URL", RBF_BINARY_URL)
        print(f"  [RBF] downloading binary from {url}")
        with urlopen(Request(url, headers={"User-Agent": "realitystream"}), timeout=120) as resp:
            data = resp.read()
        tmp_zip = os.path.join(target, f"rbf_{uuid.uuid4().hex}.zip")
        with open(tmp_zip, "wb") as fh:
            fh.write(data)
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(target)
        os.remove(tmp_zip)
        found = next((os.path.join(r, "rbf") for r, _, files in os.walk(target) if "rbf" in files), None)
        if found is None:
            raise FileNotFoundError("Downloaded zip did not contain an 'rbf' executable.")
        if os.path.abspath(found) != os.path.abspath(path):
            shutil.copy2(found, path)
        os.chmod(path, 0o755)
        return path


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(key, random_state=RANDOM_STATE):
    """Estimator for a canonical model key; cuML classes when the GPU path is on."""
    gpu = gpu_enabled()
    if key == "rfc":
        if gpu:
            from cuml.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_streams=1)
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
    if key == "lr":
        if gpu:
            from cuml.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, penalty="l2")
        from sklearn.linear_model import LogisticRegression
        # L2 is the default penalty; scikit-learn 1.8 deprecated the penalty argument (see _lr_kwargs)
        return LogisticRegression(max_iter=1000)
    if key == "svm":
        if gpu:
            from cuml.svm import SVC
            return SVC(probability=True, kernel="rbf", C=1.0)
        from sklearn.svm import SVC
        return SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
    if key == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(random_state=random_state)
    if key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", device="cuda" if gpu else "cpu",
                             eval_metric="logloss", random_state=random_state, n_jobs=-1)
    if key == "rbf":
        return RandomBitsForest()
    raise ValueError(f"Unknown model key: {key}")


def _param_grid(key, n_iter, rng):
    if key == "xgboost":
        return {
            "n_estimators": rng.integers(50, 150, n_iter).tolist(),
            "learning_rate": rng.uniform(0.01, 0.2, n_iter).tolist(),
            "max_depth": rng.integers(3, 8, n_iter).tolist(),
            "subsample": rng.uniform(0.6, 1.0, n_iter).tolist(),
            "colsample_bytree": rng.uniform(0.6, 1.0, n_iter).tolist(),
        }
    if key == "mlp":
        return {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": np.logspace(-4, -2, n_iter).tolist(),
            "learning_rate_init": rng.uniform(0.0005, 0.01, n_iter).tolist(),
            "max_iter": [300, 500],
        }
    return None


# Models whose features must be standardized before fitting (distance/gradient based).
SCALE_SENSITIVE = {"lr", "svm", "mlp"}


def train_models(X_train, y_train, X_test, y_test, keys, random_state=RANDOM_STATE, n_iter=20, smote=False,
                 hyperparameters=None, errors=None):
    """Train each requested model; return a list of result dicts (same fields as the colab, plus the test
    predictions "y_pred" and "y_score" and a "predictor" taking unscaled rows, for the charts).

    Scaling (lr/svm/mlp) and SMOTE are fit on training data only. When a hyperparameter search
    runs they are imblearn Pipeline steps (grid keys prefixed ``model__``), so each CV fold fits
    them on its own training part; otherwise they are applied to the training split directly, which
    is leak-free as there is no CV there. Returns ``[]`` when SMOTE is requested but a class has
    fewer than 2 rows.

    hyperparameters: {model_key: {field: value}} from validate_parameters(). Fixed values are set on the
    estimator and leave the search grid. Each result's "params_used" lists the model's settings.
    errors: a dict that receives {model_key: reason} for each model that couldn't be trained.
    """
    errors = {} if errors is None else errors
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = X_train.fillna(0), X_test.fillna(0)
    y_train_np, y_test_np = np.asarray(y_train).ravel(), np.asarray(y_test).ravel()
    class_counts = pd.Series(y_train_np).value_counts()
    if smote and (len(class_counts) < 2 or class_counts.min() < 2):
        print("  [WARN] SMOTE needs two classes with at least 2 samples each; skipping this pass.")
        return []
    can_search = len(class_counts) > 1 and class_counts.min() >= 5
    binary = len(class_counts) == 2
    smote_k = min(5, int(class_counts.min()) - 1) if smote else None
    # In the search, SMOTE sees one CV fold's training part, 4/5 of the rows, so its smallest class is smaller
    search_smote_k = min(5, int(class_counts.min()) * 4 // 5 - 1) if smote else None
    rng = np.random.default_rng(random_state)
    gpu = gpu_enabled()
    results = []

    for key in keys:
        fixed = (hyperparameters or {}).get(key) or {}
        try:
            model = apply_hyperparameters(make_model(key, random_state), key, fixed, gpu)
        except Exception as exc:
            print(f"  [WARN] Skipping {key}: {exc}")
            errors[key] = str(exc)
            continue
        print(f"\n[MODEL] Training {MODEL_TITLES[key]} ({key})...")
        scale = key in SCALE_SENSITIVE
        start = time.time()
        try:
            # The grid is drawn before fixed fields leave it, so other fields get the same candidates
            grid = {k: v for k, v in (_param_grid(key, n_iter, rng) or {}).items() if k not in fixed}
            if grid and can_search:
                # Scaling and SMOTE as pipeline steps: each CV fold fits them on its own training part.
                steps = []
                if scale:
                    steps.append(("scaler", StandardScaler()))
                if smote:
                    steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=search_smote_k)))
                steps.append(("model", model))
                pipe = ImbPipeline(steps)
                search = RandomizedSearchCV(
                    pipe, param_distributions={f"model__{k}": v for k, v in grid.items()}, n_iter=n_iter,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                    scoring="roc_auc" if binary else "accuracy", n_jobs=-1, random_state=random_state,
                )
                search.fit(X_train, y_train_np)
                fitted = predictor = search.best_estimator_
                best_model = fitted.named_steps["model"]  # final estimator, so feature_importances still works
                y_pred = np.asarray(to_cpu(fitted.predict(X_test))).ravel()
                y_prob = np.asarray(to_cpu(fitted.predict_proba(X_test))) if hasattr(fitted, "predict_proba") else None
            else:
                # No search, so no CV here: scale and SMOTE the training split directly (leak-free).
                X_tr, X_te, y_tr = X_train, X_test, y_train_np
                predictor = model
                if scale:
                    from sklearn.pipeline import Pipeline

                    scaler = StandardScaler().set_output(transform="pandas")  # Keeps the column names
                    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_train.columns, index=X_train.index)
                    X_te = pd.DataFrame(scaler.transform(X_te), columns=X_test.columns, index=X_test.index)
                    predictor = Pipeline([("scaler", scaler), ("model", model)])  # Both fitted, for unscaled rows
                if smote:
                    X_tr, y_tr = SMOTE(random_state=random_state, k_neighbors=smote_k).fit_resample(X_tr, y_tr)
                if gpu and key in ("rfc", "lr", "svm"):
                    import cudf
                    import cupy as cp
                    model.fit(cudf.DataFrame.from_pandas(pd.DataFrame(X_tr).reset_index(drop=True)),
                              cp.asarray(np.asarray(y_tr)))
                    X_te = cudf.DataFrame.from_pandas(X_te)
                    predictor = None  # Takes cudf rows, so no permutation importance
                else:
                    model.fit(X_tr, y_tr)
                best_model = model
                y_pred = np.asarray(to_cpu(model.predict(X_te))).ravel()
                y_prob = np.asarray(to_cpu(model.predict_proba(X_te))) if hasattr(model, "predict_proba") else None
        except Exception as exc:
            print(f"  [WARN] {key} failed: {exc}")
            errors[key] = str(exc)
            continue
        elapsed = time.time() - start

        report = classification_report(y_test_np, y_pred, output_dict=True, zero_division=0)
        two_classes = len(np.unique(y_test_np)) > 1
        roc = roc_auc_score(y_test_np, y_prob[:, 1]) if (y_prob is not None and two_classes) else None
        pos = report.get("1", {})
        gmean = (report["0"]["recall"] * report["1"]["recall"]) ** 0.5 if ("0" in report and "1" in report) else 0.0
        result = {
            "model_type": key,
            "best_model": best_model,
            "accuracy": round(accuracy_score(y_test_np, y_pred), 4),
            "roc_auc": None if roc is None else round(roc, 4),
            "gmean": round(gmean, 4),
            "precision": round(pos.get("precision", 0.0), 4),
            "recall": round(pos.get("recall", 0.0), 4),
            "f1_score": round(pos.get("f1-score", 0.0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test_np, y_pred), 4),
            "time": round(elapsed, 2),
            "params_used": params_used(key, best_model),
            "classification_report": report,
            "y_pred": y_pred,
            "y_score": y_prob[:, 1] if (y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2) else None,
            "predictor": predictor,
        }
        print(f"  Accuracy {result['accuracy']}  ROC-AUC {result['roc_auc']}  F1 {result['f1_score']}  "
              f"Balanced-Acc {result['balanced_accuracy']}  ({result['time']}s)")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

_NAICS_NAMES = {}


def _naics_table(digits):
    """{code: name} for 2- or 6-digit NAICS, loaded once, failsafe."""
    if digits not in _NAICS_NAMES:
        try:
            if digits == 6:
                df = pd.read_excel(NAICS6_NAMES_URL, dtype=str, skiprows=1, usecols=[0, 1])
            else:
                df = pd.read_csv(NAICS2_NAMES_URL, dtype=str)
            df.columns = ["code", "name"]
            df["name"] = df["name"].str.strip()
            _NAICS_NAMES[digits] = df.set_index("code")["name"].to_dict()
        except Exception as exc:
            print(f"  [WARN] NAICS{digits} names unavailable ({exc}); keeping raw codes.")
            _NAICS_NAMES[digits] = {}
    return _NAICS_NAMES[digits]


def naics_name(feature):
    """Emp-11 -> 'Emp-11-Agriculture, Forestry, Fishing and Hunting'; Emp-454310-2019 -> 'Emp-454310-Fuel Dealers-2019'.

    Prefix (Emp/Est/Pay) and year are kept so names stay unique and recoverable."""
    match = re.match(r"^(Emp|Est|Pay)-(\d{2}|\d{6})(?:-(\d{4}))?$", str(feature))
    if not match:
        return feature
    prefix, code, year = match.groups()
    name = _naics_table(len(code)).get(code)
    if not name:
        return feature
    return f"{prefix}-{code}-{name}" + (f"-{year}" if year else "")


PERMUTATION_MODELS = ("svm", "mlp")  # No built-in importances. rbf retrains on each prediction, so it's left out
PERMUTATION_ROWS = 1000  # Test rows shuffled for permutation importance
PERMUTATION_FEATURES = 200  # Wider data is skipped: each feature is scored 5 times


def feature_importances(results, feature_names, map_naics=True, X_test=None, y_test=None):
    """{model_key: DataFrame(Feature, Importance)} for models that expose importances, plus svm and mlp by
    permutation on the test rows when X_test and y_test are given (DataFrame.attrs["method"] is "permutation")."""
    out = {}
    for r in results:
        key, model = r["model_type"], r["best_model"]
        method = ""
        if key == "xgboost":
            scores = model.get_booster().get_score(importance_type="weight")
            values = [scores.get(f, scores.get(f"f{i}", 0)) for i, f in enumerate(feature_names)]
        elif key == "rfc" and hasattr(model, "feature_importances_"):
            values = np.asarray(to_cpu(model.feature_importances_)).ravel()
        elif key == "lr" and hasattr(model, "coef_"):
            values = np.abs(np.asarray(to_cpu(model.coef_))).ravel()
        elif key in PERMUTATION_MODELS and X_test is not None and r.get("predictor") is not None:
            values = _permutation_values(r["predictor"], X_test, y_test)
            if values is None:
                continue
            method = "permutation"
        else:
            continue
        df = pd.DataFrame({"Feature": list(feature_names), "Importance": values})
        if map_naics:
            df["Feature"] = df["Feature"].map(naics_name)
        df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
        df.attrs["method"] = method
        out[key] = df
    return out


def _permutation_values(predictor, X_test, y_test):
    """Mean drop in ROC-AUC (accuracy for more than two classes) when each feature is shuffled, or None."""
    from sklearn.inspection import permutation_importance

    if X_test.shape[1] > PERMUTATION_FEATURES:
        print(f"  [INFO] No permutation importance for {X_test.shape[1]} features (limit {PERMUTATION_FEATURES})")
        return None
    X = X_test.fillna(0)  # Same imputation as train_models
    y = np.asarray(y_test).ravel()
    if len(X) > PERMUTATION_ROWS:
        rows = np.random.default_rng(RANDOM_STATE).choice(len(X), PERMUTATION_ROWS, replace=False)
        X, y = X.iloc[rows], y[rows]
    if len(np.unique(y)) < 2:
        return None
    scoring = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
    try:
        return permutation_importance(predictor, X, y, scoring=scoring, n_repeats=5, random_state=RANDOM_STATE,
                                      n_jobs=-1).importances_mean
    except Exception as exc:
        print(f"  [WARN] Permutation importance failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Report folder and upload
# ---------------------------------------------------------------------------

REPORT_COLUMNS = ["Model", "Accuracy", "ROC_AUC", "F1_Score", "Precision", "Recall", "GMean",
                  "Training_Time_Seconds", "Balanced_Accuracy", "Lift_Over_Baseline"]


def results_table(results):
    return pd.DataFrame([{
        "Model": r["model_type"], "Accuracy": r["accuracy"], "ROC_AUC": r["roc_auc"],
        "F1_Score": r["f1_score"], "Precision": r["precision"], "Recall": r["recall"],
        "GMean": r["gmean"], "Training_Time_Seconds": r["time"],
        "Balanced_Accuracy": r.get("balanced_accuracy"), "Lift_Over_Baseline": r.get("lift_over_baseline"),
    } for r in results], columns=REPORT_COLUMNS)


def setup_report_folder(report_dir):
    """Fresh report folder with index.html (localsite template), README.md and model-options.csv."""
    if os.path.isdir(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)
    index = os.path.join(report_dir, "index.html")
    try:
        resp = requests.get(REPORT_TEMPLATE_URL, timeout=60)
        resp.raise_for_status()
        with open(index, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
    except Exception as exc:
        print(f"  [WARN] Could not download report template: {exc}")
    with open(os.path.join(report_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Run Models Report\n\nThis folder contains generated reports from model executions.")
    with open(os.path.join(report_dir, "model-options.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_name"])
        for name in ("LR", "RFC", "RBF", "SVM", "MLP", "XGBoost"):
            writer.writerow([name])


def report_readme(params, results_no_smote, results_smote, errors=None, rows=None, charted="", classes=None,
                  smote_skipped=False, warnings=None, failure=None):
    """README.md for a run's report folder: what ran, which pass the charts show, and why any model wasn't trained.
    classes: {target value: row count}. smote_skipped: the SMOTE pass was asked for but had too few rows.
    warnings: data warnings from warn(). failure: the error that stopped the run before training."""
    lines = ["# Run Models Report", ""]
    if failure:
        lines += [f"The run stopped before training: {failure}", ""]
    about = f"Parameters: {params.get('folder') or 'unnamed'}"
    if rows:
        about += f", {rows[0]:,} rows and {rows[1]:,} features"
    if classes:
        about += ". Target rows: " + ", ".join(f"{value}: {count:,}" for value, count in sorted(classes.items()))
    lines.append(about + ".")
    passes = [label for label, results in (("without SMOTE", results_no_smote), ("with SMOTE", results_smote))
              if results]
    trained = list(dict.fromkeys(r["model_type"] for r in results_no_smote + results_smote))
    if trained:
        lines.append(f"Trained {', '.join(MODEL_TITLES[k] for k in trained)} ({' and '.join(passes)})."
                     + (f" The charts show the run {charted}." if charted else ""))
    elif not params.get("models") and not failure:
        lines.append("No models were selected.")
    if smote_skipped:
        lines.append("SMOTE was skipped: it needs two classes with at least 2 training rows each.")
    if warnings:
        lines += ["", "Data warnings:", ""] + [f"- {msg}" for msg in warnings]
    if errors:
        lines += ["", "Not trained:", ""] + [f"- {MODEL_TITLES.get(k, k)}: {msg}" for k, msg in errors.items()]
    return "\n".join(lines) + "\n"


def write_reports(report_dir, params, results_no_smote, results_smote, importances, readme=None):
    if readme:
        with open(os.path.join(report_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme)
    with open(os.path.join(report_dir, "parameters.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh, sort_keys=False)
    if results_no_smote:
        results_table(results_no_smote).to_csv(os.path.join(report_dir, "model_performance_report_no_smote.csv"), index=False)
    if results_smote:
        results_table(results_smote).to_csv(os.path.join(report_dir, "model_performance_report_smote.csv"), index=False)
    if "xgboost" in importances:
        importances["xgboost"].to_csv(os.path.join(report_dir, "feature_importance_xgboost.csv"), index=False)
    print(f"\n[REPORT] {len(os.listdir(report_dir))} files in {os.path.abspath(report_dir)}")


def report_name(now=None):
    """Report folder path for a run started now: {year}/run-{date}T{time}, as in the reports repo."""
    now = now or datetime.now()
    return now.strftime("%Y/run-%Y-%m-%dT%H-%M-%S")


def write_file_list(report_dir):
    """files.json: the report folder's files, which its index.html (localsite's displayCharts) shows."""
    files = sorted(p.relative_to(report_dir).as_posix() for p in Path(report_dir).glob("**/*")
                   if p.is_file() and p.name not in ("index.html", "files.json"))
    with open(os.path.join(report_dir, "files.json"), "w", encoding="utf-8") as fh:
        json.dump({"files": files}, fh, indent=2)


def upload_reports(report_dir, repo=None, branch="main", year=None, subfolder=None, token=None):
    """Commit every file in report_dir to {year}/{subfolder}/ in the reports repo. Returns that path."""
    repo = repo or reports_repo()
    token = token or require_env("GITHUB_REPORTS_TOKEN")
    default_year, default_subfolder = report_name().split("/")
    year = year or default_year
    subfolder = subfolder or default_subfolder
    remote_dir = f"{year}/{subfolder}"
    api = f"https://api.github.com/repos/{repo}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    def call(method, url, **kw):
        resp = requests.request(method, url, headers=headers, timeout=60, **kw)
        resp.raise_for_status()
        return resp.json()

    head_sha = call("GET", f"{api}/git/refs/heads/{branch}")["object"]["sha"]
    base_tree = call("GET", f"{api}/git/commits/{head_sha}")["tree"]["sha"]
    tree = []
    for path in sorted(Path(report_dir).glob("**/*")):
        if path.is_file():
            data = path.read_bytes()
            try:
                entry = {"content": data.decode("utf-8")}
            except UnicodeDecodeError:  # Charts and other binary files go up as base64 blobs
                blob = call("POST", f"{api}/git/blobs",
                            json={"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"})
                entry = {"sha": blob["sha"]}
            tree.append({"path": f"{remote_dir}/{path.relative_to(report_dir).as_posix()}",
                         "mode": "100644", "type": "blob", **entry})
    new_tree = call("POST", f"{api}/git/trees", json={"base_tree": base_tree, "tree": tree})["sha"]
    commit = call("POST", f"{api}/git/commits",
                  json={"message": f"Run Models report {subfolder}", "tree": new_tree, "parents": [head_sha]})["sha"]
    call("PATCH", f"{api}/git/refs/heads/{branch}", json={"sha": commit})
    print(f"[UPLOAD] {len(tree)} files -> https://github.com/{repo}/tree/{branch}/{remote_dir}")
    return remote_dir


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

NOT_SUMMARIZED = {"best_model", "predictor", "y_pred", "y_score"}  # Estimators and arrays, for the charts


def summarize(results):
    """JSON-safe view of a results list (drops the fitted estimators and test predictions)."""
    return [{k: v for k, v in r.items() if k not in NOT_SUMMARIZED} for r in results]


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Train/test split: stratified when every class has >= 2 rows, else unstratified with a
    warning. Also warns on a tiny dataset (< 100 rows) and on a single-class test set."""
    from sklearn.model_selection import train_test_split

    if len(X) < 100:
        warn(f"Only {len(X)} rows; model estimates will be unstable.")
    counts = pd.Series(np.asarray(y).ravel()).value_counts()
    if len(counts) < 2:
        warn(f"Every row has target {counts.index[0]}, so models have only one class to learn.")
    elif int(counts.min()) < 5:
        warn(f"Only {int(counts.min())} of {len(y)} rows have target {counts.idxmin()}. Most models need more of "
             "each class: add rows, for example more states or years.")
    stratify = y if (len(counts) >= 2 and int(counts.min()) >= 2) else None
    if stratify is None and len(counts) >= 2:
        warn("Not every class has 2 rows or more, so the train/test split isn't stratified.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    if len(counts) >= 2 and len(np.unique(np.asarray(y_train).ravel())) < 2:
        warn(f"The {len(y_train)} training rows have only one class, so most models can't be trained.")
    if len(np.unique(np.asarray(y_test).ravel())) < 2:
        warn("The test rows have a single class, so ROC-AUC is undefined and reported as None.")
    return X_train, X_test, y_train, y_test


def majority_baseline(y_train, y_test):
    """Scores for always predicting the training-majority class: accuracy, balanced accuracy, class."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    y_train_np, y_test_np = np.asarray(y_train).ravel(), np.asarray(y_test).ravel()
    majority = pd.Series(y_train_np).value_counts().idxmax()
    pred = np.full(len(y_test_np), majority)
    return {
        "accuracy": round(accuracy_score(y_test_np, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test_np, pred), 4),
        "majority_class": int(majority) if np.issubdtype(np.asarray(majority).dtype, np.number) else str(majority),
    }


def _apply_lift(results, baseline):
    """Set lift_over_baseline (balanced accuracy minus the baseline's) on each result; warn when <= 0."""
    for r in results:
        lift = round(r["balanced_accuracy"] - baseline["balanced_accuracy"], 4)
        r["lift_over_baseline"] = lift
        if lift <= 0:
            print(f"  [WARN] {r['model_type']} balanced-accuracy lift over baseline is {lift} (<= 0).")


CV_METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1_macro")
CV_REPORT_COLUMNS = ["Model", "ROC_AUC_Mean", "ROC_AUC_Std", "PR_AUC_Mean", "PR_AUC_Std",
                     "Balanced_Accuracy_Mean", "Balanced_Accuracy_Std", "F1_Macro_Mean", "F1_Macro_Std",
                     "Lift_Over_Baseline"]


def cv_results_table(results):
    return pd.DataFrame([{
        "Model": r["model"], "ROC_AUC_Mean": r["roc_auc_mean"], "ROC_AUC_Std": r["roc_auc_std"],
        "PR_AUC_Mean": r["pr_auc_mean"], "PR_AUC_Std": r["pr_auc_std"],
        "Balanced_Accuracy_Mean": r["balanced_accuracy_mean"], "Balanced_Accuracy_Std": r["balanced_accuracy_std"],
        "F1_Macro_Mean": r["f1_macro_mean"], "F1_Macro_Std": r["f1_macro_std"],
        "Lift_Over_Baseline": r.get("lift_over_baseline"),
    } for r in results], columns=CV_REPORT_COLUMNS)


def _cv_fold_scores(estimator, X_tr, y_tr, X_te, y_te):
    """Fit on a fold's training rows, score on its held-out rows. ROC/PR are nan if the fold is single-class."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

    estimator.fit(X_tr, y_tr)
    y_pred = np.asarray(to_cpu(estimator.predict(X_te))).ravel()
    scores = {"roc_auc": np.nan, "pr_auc": np.nan,
              "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
              "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0)}
    classes = getattr(estimator, "classes_", np.unique(y_tr))
    if len(np.unique(y_te)) > 1 and len(classes) == 2 and hasattr(estimator, "predict_proba"):
        proba = np.asarray(to_cpu(estimator.predict_proba(X_te)))[:, 1]
        scores["roc_auc"] = roc_auc_score(y_te, proba)
        scores["pr_auc"] = average_precision_score(y_te, proba)
    return scores


def _cv_pipeline(key, random_state, smote_k, fixed=None):
    """Fixed-setting estimator for CV: scaler (lr/svm/mlp) + optional SMOTE + model, no nested search."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler

    steps = []
    if key in SCALE_SENSITIVE:
        steps.append(("scaler", StandardScaler()))
    if smote_k is not None:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=smote_k)))
    steps.append(("model", apply_hyperparameters(make_model(key, random_state), key, fixed, gpu_enabled())))
    return ImbPipeline(steps)


def cross_validate_models(X, y, groups, keys, folds=5, cv="stratified", random_state=RANDOM_STATE,
                          hyperparameters=None):
    """Cross-validate with fixed model settings (no nested search). Scaling and SMOTE are pipeline steps
    fit per fold, so they never see held-out rows. cv is 'stratified' (StratifiedKFold over all rows) or
    'group' (GroupKFold by state, falling back to stratified when there are fewer groups than folds).
    Returns (results, cv_used): per-model fold mean/std for each metric plus baseline lift."""
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    X = X.reset_index(drop=True).fillna(0)   # same imputation as train_models
    y_np = np.asarray(y).ravel()
    cv = str(cv).lower()
    n_groups = len(pd.unique(pd.Series(groups).dropna())) if groups is not None else 0
    if cv == "group" and n_groups < folds:
        warn(f"Group cross-validation needs {folds} groups or more (have {n_groups}), so it's stratified instead.")
        cv = "stratified"
    if cv == "group":
        splits = list(GroupKFold(n_splits=folds).split(X, y_np, np.asarray(pd.Series(groups).astype("float"))))
    else:
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state).split(X, y_np))

    def aggregate(name, make_estimator, use_smote):
        fold_rows = []
        for tr, te in splits:
            y_tr = y_np[tr]
            smote_k = None
            if use_smote:
                counts = pd.Series(y_tr).value_counts()
                if len(counts) >= 2 and counts.min() >= 2:
                    smote_k = min(5, int(counts.min()) - 1)
            fold_rows.append(_cv_fold_scores(make_estimator(smote_k), X.iloc[tr], y_tr, X.iloc[te], y_np[te]))
        agg = {"model": name}
        for metric in CV_METRICS:
            vals = np.array([f[metric] for f in fold_rows], dtype=float)
            allnan = np.isnan(vals).all()
            agg[f"{metric}_mean"] = None if allnan else round(float(np.nanmean(vals)), 4)
            agg[f"{metric}_std"] = None if allnan else round(float(np.nanstd(vals)), 4)
        return agg

    results = [aggregate("MajorityBaseline", lambda sk: DummyClassifier(strategy="most_frequent"), use_smote=False)]
    results[0]["lift_over_baseline"] = 0.0
    base = results[0]["balanced_accuracy_mean"] or 0.0
    print(f"\n[CV {cv}, {folds} folds] MajorityBaseline  balanced_acc {results[0]['balanced_accuracy_mean']}")
    for key in keys:
        try:
            make_model(key, random_state)
        except Exception as exc:
            print(f"  [WARN] Skipping {key}: {exc}")
            continue
        try:
            fixed = (hyperparameters or {}).get(key)
            agg = aggregate(key, lambda sk, key=key, fixed=fixed: _cv_pipeline(key, random_state, sk, fixed),
                            use_smote=True)
        except Exception as exc:
            print(f"  [WARN] {key} failed: {exc}")
            continue
        agg["lift_over_baseline"] = round((agg["balanced_accuracy_mean"] or 0.0) - base, 4)
        results.append(agg)
        print(f"  {key}: roc_auc {agg['roc_auc_mean']}±{agg['roc_auc_std']}  pr_auc {agg['pr_auc_mean']}±{agg['pr_auc_std']}  "
              f"bal_acc {agg['balanced_accuracy_mean']}±{agg['balanced_accuracy_std']}  "
              f"f1 {agg['f1_macro_mean']}±{agg['f1_macro_std']}  lift {agg['lift_over_baseline']}")
    return results, cv


def _warn_timeseries(param):
    """The YAML's timeseries section isn't used yet; say so, and whether its file is reachable."""
    ts_path = getattr(getattr(param, "timeseries", None), "path", None)
    if not ts_path:
        return
    try:
        reachable = requests.head(ts_path, timeout=30, allow_redirects=True).status_code == 200
        state = "reachable" if reachable else "not found"
    except Exception as exc:
        state = f"not checked ({exc.__class__.__name__})"
    warn(f"timeseries.path is {state} and not used yet: {ts_path}")


# How to give a token write access, for the no-access messages
NO_WRITE_FIX = ('In the token\'s settings on GitHub, set Repository permissions, "Contents" to Read and write. '
                "A token for an organization's repo may also need an organization owner's approval.")


def check_reports_token(token=None, label="GITHUB_REPORTS_TOKEN"):
    """Whether GITHUB_REPORTS_TOKEN can push to the reports repo, asked of GitHub. Returns
    {"token": "valid" | "missing" | "invalid" | "no_access" | "unknown", "message": text for the page}. Never the value.
    label names the token in messages, for one sent from the browser."""
    token = token if token is not None else get_env("GITHUB_REPORTS_TOKEN")
    if not token:
        return {"token": "missing", "message": f"{label} is not set, so output isn't pushed to GitHub."}
    repo = reports_repo()
    # The repo's permissions.push is the account's role, not a fine-grained token's own access, so ask for
    # a branch at a commit that can't exist: GitHub checks write access first (403), then the commit (422)
    try:
        resp = requests.post(f"https://api.github.com/repos/{repo}/git/refs", timeout=15,
                             headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
                             json={"ref": "refs/heads/rs-token-check", "sha": "0" * 40})
    except Exception:
        return {"token": "unknown", "message": f"GitHub could not be reached to check {label}."}
    if resp.status_code == 401:
        return {"token": "invalid", "message": f"GitHub rejects {label}: it has expired or been revoked (401), "
                                                "so output won't be pushed to GitHub."}
    if resp.status_code == 422:
        return {"token": "valid", "message": f"{label} works: output will be pushed to {repo}."}
    if resp.status_code in (403, 404):
        return {"token": "no_access", "message": f"{label} can't write to {repo}, so output won't be "
                                                  "pushed to GitHub. " + NO_WRITE_FIX}
    return {"token": "unknown", "message": f"GitHub answered HTTP {resp.status_code} when checking {label}."}


def push_report(report_dir, upload, name=None, token=None, label="GITHUB_REPORTS_TOKEN"):
    """Push the report folder to the reports repo. upload: True (GITHUB_REPORTS_TOKEN required), False
    (don't), or "auto" (push when the token is available). Returns (uploaded_to or None, github status):
    {"token": "available" | "missing" | None, "pushed_to": path or None, "message": text for the page},
    plus "repo" and "view_url" (the report on the repo's website) after a push.
    token: one sent from the browser, used in place of the setting; label names it in messages."""
    if not upload:
        return None, {"token": None, "pushed_to": None, "message": ""}
    token = token or get_env("GITHUB_REPORTS_TOKEN")
    if not token:
        if upload != "auto":
            raise MissingKey("GITHUB_REPORTS_TOKEN")
        return None, {"token": "missing", "pushed_to": None,
                      "message": f"{label} is not set, so the report wasn't pushed to GitHub."}
    year, subfolder = (name or report_name()).split("/")
    repo = reports_repo()
    try:
        pushed = upload_reports(report_dir, repo=repo, year=year, subfolder=subfolder, token=token)
    except Exception as exc:
        print(f"  [WARN] GitHub push failed: {exc}")
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            message = (f"GitHub rejected {label}: it has expired or been revoked (401). "
                       "Create a new one at https://github.com/settings/tokens and replace it.")
        elif status in (403, 404):
            message = f"{label} can't write to {repo} (HTTP {status}). " + NO_WRITE_FIX
        else:
            message = f"{label} is set, but the push to GitHub failed: {exc}"
        return None, {"token": "invalid" if status == 401 else "available", "pushed_to": None, "message": message}
    view_url = f"{reports_site(repo)}{pushed}/"
    return pushed, {"token": "available", "pushed_to": pushed, "repo": repo, "view_url": view_url,
                    "message": f"{label} is set. View at {view_url}. Pushed to {repo}/{pushed}."}


def run_pipeline(yaml_path, report_dir="report", upload=False, n_iter=None, smote=None, name=None,
                 token=None, token_label="GITHUB_REPORTS_TOKEN"):
    """Load params -> fetch data -> train (plain and SMOTE) -> write report folder -> optional upload.

    smote: None trains both without and with SMOTE, False only without, True only with.
    upload: True, False or "auto" (push_report). name: the {year}/run-{date-time} folder it pushes to.
    token: a GitHub token sent from the browser, pushed with in place of GITHUB_REPORTS_TOKEN (token_label names it).
    With `models: []` no model is trained, and the report folder still has the parameters and baseline.
    An optional ``evaluation:`` block in the YAML adds k-fold cross-validation (``model_performance_cv.csv``
    and the "cv" return key); the existing report files and behaviour are unchanged when it is absent.
    Optional ``split``, ``search`` and ``hyperparameters`` sections (api/model-params.yaml) set the test
    share and seed, the search size, and model settings. n_iter, when given, overrides search.n_iter.
    """
    print("=" * 60 + "\n  RealityStream Run Models\n" + "=" * 60)
    params = load_parameters(yaml_path)
    param = DictToObject(OrderedDict(params))
    keys = params["models"]
    print(f"[PARAMS] {yaml_path}\n   folder: {params.get('folder', 'N/A')}   models: {keys}   gpu: {gpu_enabled()}")
    settings = validate_parameters(params)
    schema = model_params_schema()
    test_size = settings["split"].get("test_size", schema["split"]["test_size"]["default"])
    seed = settings["split"].get("random_state", schema["split"]["random_state"]["default"])
    if n_iter is None:
        n_iter = settings["search"].get("n_iter", schema["search"]["n_iter"]["default"])
    hyper = settings["hyperparameters"]
    print(f"   test_size: {test_size}   random_state: {seed}   n_iter: {n_iter}"
          + "".join(f"\n   {k}: {v}" for k, v in hyper.items() if k in keys))

    warnings = []  # Data warnings (warn()), listed in README.md and the reply
    _RUN_WARNINGS.set(warnings)
    evaluation = params.get("evaluation") or {}
    print("\n[DATA] Loading...")
    try:
        if evaluation:
            X, y, groups = load_data(param, return_groups=True)
        else:
            X, y = load_data(param)
            groups = None
        _warn_timeseries(param)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=seed)
    except Exception as exc:
        # The report folder still records the parameters and why the data couldn't be used
        setup_report_folder(report_dir)
        write_reports(report_dir, params, [], [], {},
                      readme=report_readme(params, [], [], warnings=warnings, failure=f"{exc.__class__.__name__}: {exc}"))
        write_file_list(report_dir)
        try:
            push_report(report_dir, upload, name, token=token, label=token_label)
        except MissingKey:
            pass
        raise
    print(f"  Train: {len(X_train)} rows   Test: {len(X_test)} rows   Features: {X.shape[1]}")

    baseline = majority_baseline(y_train, y_test)
    print(f"  Baseline (predict {baseline['majority_class']}): accuracy {baseline['accuracy']}  "
          f"balanced_acc {baseline['balanced_accuracy']}")

    results_no_smote, results_smote, errors_no_smote, errors_smote = [], [], {}, {}
    if not keys:
        print("\n[TRAIN] No models selected, so none are trained")
    if keys and smote is not True:
        print("\n[TRAIN] Without SMOTE")
        results_no_smote = train_models(X_train, y_train, X_test, y_test, keys, random_state=seed, n_iter=n_iter,
                                        smote=False, hyperparameters=hyper, errors=errors_no_smote)
        _apply_lift(results_no_smote, baseline)

    if keys and smote is not False:
        print("\n[TRAIN] With SMOTE")
        results_smote = train_models(X_train, y_train, X_test, y_test, keys, random_state=seed, n_iter=n_iter,
                                     smote=True, hyperparameters=hyper, errors=errors_smote)
        _apply_lift(results_smote, baseline)
    smote_skipped = bool(keys) and smote is not False and not results_smote and not errors_smote
    # {model: reason} for each model a pass couldn't train, naming the pass when the other one trained it
    trained = {r["model_type"] for r in results_no_smote + results_smote}
    errors = {}
    for label, failed in (("without SMOTE", errors_no_smote), ("with SMOTE", errors_smote)):
        for k, msg in failed.items():
            errors.setdefault(k, f"{msg} (failed {label}; the other pass trained it)" if k in trained else msg)

    # The charts show one pass: the one that trained more models, with SMOTE on a tie
    with_smote = len(results_smote) >= len(results_no_smote) and bool(results_smote)
    charted = results_smote if with_smote else results_no_smote
    charted_label = "with SMOTE" if with_smote else "without SMOTE" if results_no_smote else ""
    importances = feature_importances(charted, list(X.columns), X_test=X_test, y_test=y_test)

    setup_report_folder(report_dir)
    classes = {_json_value(k): int(v) for k, v in pd.Series(np.asarray(y).ravel()).value_counts().items()}
    readme = report_readme(params, results_no_smote, results_smote, errors, rows=X.shape, charted=charted_label,
                           classes=classes, smote_skipped=smote_skipped, warnings=warnings)
    write_reports(report_dir, params, results_no_smote, results_smote, importances, readme=readme)
    from charts import write_charts

    write_charts(report_dir, charted, y_test, importances, titles=MODEL_TITLES, label=charted_label)

    cv = None
    if evaluation and keys:
        folds = int(evaluation.get("folds", 5))
        cv_results, cv_used = cross_validate_models(X, y, groups, keys, folds=folds,
                                                    cv=evaluation.get("cv", "stratified"),
                                                    random_state=seed, hyperparameters=hyper)
        cv_results_table(cv_results).to_csv(os.path.join(report_dir, "model_performance_cv.csv"), index=False)
        cv = {"mode": cv_used, "folds": folds, "results": cv_results}

    write_file_list(report_dir)
    uploaded, github = push_report(report_dir, upload, name, token=token, label=token_label)
    print("\n[DONE]")
    summary = {
        "folder": params.get("folder"),
        "report_dir": os.path.abspath(report_dir),
        "baseline": baseline,
        "settings": {"test_size": test_size, "random_state": seed, "n_iter": n_iter},
        "no_smote": summarize(results_no_smote),
        "smote": summarize(results_smote),
        "feature_importance": {k: v.head(20).to_dict(orient="records") for k, v in importances.items()},
        "errors": errors,
        "warnings": warnings,
        "uploaded_to": uploaded,
        "github": github,
    }
    if cv is not None:
        summary["cv"] = cv
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run RealityStream models from a parameters.yaml",
                                     epilog="Model keys: lr, rfc, rbf, svm, mlp, xgboost")
    parser.add_argument("yaml", nargs="?", default=os.environ.get("PARAMETERS_YAML_PATH"),
                        help="parameters.yaml path or URL (default: $PARAMETERS_YAML_PATH, else the first "
                             "entry of parameters/parameter-paths.csv, as in the colab)")
    parser.add_argument("--report-dir", default="report", help="output folder (default: report)")
    parser.add_argument("--upload", action="store_true", help="push the report folder to modelearth/reports")
    parser.add_argument("--n-iter", type=int, help="RandomizedSearchCV iterations for xgboost/mlp "
                                                   "(default: search.n_iter in the YAML, else 20)")
    parser.add_argument("--smote", choices=["0", "1"], help="0 = only without SMOTE, 1 = only with SMOTE (default: both)")
    args = parser.parse_args()
    smote = None if args.smote is None else args.smote == "1"
    try:
        summary = run_pipeline(args.yaml, args.report_dir, args.upload, args.n_iter, smote=smote)
    except MissingKey as exc:
        print(f"\n[KEY] {exc}")
        sys.exit(2)
    except ParameterError as exc:
        print(f"\n[PARAMS] {exc}")
        sys.exit(2)
    with open(os.path.join(args.report_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
