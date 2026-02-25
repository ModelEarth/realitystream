"""
RealityStream CLI – Run ML models from a parameters.yaml file.

Usage:
    python run_models.py parameters/parameters.yaml
    python run_models.py parameters/parameters-blinks.yaml
    python run_models.py --help

This replaces the Colab notebook workflow with a local/Cloud Run-friendly
Python script.  All heavy imports (sklearn, xgboost, …) are deferred so that
``--help`` stays fast.
"""

import argparse
import csv
import json
import os
import sys
import textwrap
import time

import pandas as pd
import requests
import yaml
from collections import OrderedDict
from io import StringIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DictToObject:
    """Recursively convert a dict to an object with dot-notation access."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, DictToObject(v) if isinstance(v, dict) else v)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, DictToObject) else v
            for k, v in vars(self).items()
        }

    def __repr__(self):
        from pprint import pformat
        body = pformat(self.to_dict(), indent=2, width=80)
        return f"DictToObject(\n{body}\n)"


def _get_common_join_column(param):
    """Return the column name used to join features ↔ targets."""
    if hasattr(param, "features") and hasattr(param.features, "common") and param.features.common:
        return param.features.common
    if hasattr(param, "targets") and hasattr(param.targets, "common") and param.targets.common:
        return param.targets.common
    if hasattr(param, "common") and param.common:
        return param.common
    return "Fips"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parameters(yaml_path: str) -> dict:
    """Load and return the YAML parameters dict."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh) or {}
    # Normalise models to a list
    models = params.get("models", [])
    if isinstance(models, str):
        models = [models]
    params["models"] = models
    return params


def _build_feature_urls(param) -> list[str]:
    """Expand the features URL template into concrete URLs."""
    template = param.features.path
    if not template:
        return []

    # Direct URL (no placeholders)
    if "{" not in template:
        return [template]

    naics_values = getattr(param.features, "naics", [])
    startyear = getattr(param.features, "startyear", None)
    endyear = getattr(param.features, "endyear", None)
    states_raw = getattr(param.features, "state", "")

    if isinstance(states_raw, list):
        states = states_raw
    elif states_raw:
        states = [s.strip() for s in str(states_raw).split(",")]
    else:
        states = []

    years = range(startyear, endyear + 1) if startyear and endyear else []

    urls: list[str] = []
    for state in (states or [""]):
        for year in (years or [0]):
            for naics in (naics_values or [0]):
                try:
                    urls.append(template.format(naics=naics, year=year, state=state))
                except KeyError:
                    pass
    return urls


def fetch_csv(url: str) -> pd.DataFrame:
    """Download a CSV from *url* and return a DataFrame."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def load_data(param):
    """
    Fetch feature + target data described by *param* and return
    (X_train, X_test, y_train, y_test, feature_names).
    """
    from sklearn.model_selection import train_test_split

    # --- Features -----------------------------------------------------------
    feature_urls = _build_feature_urls(param)
    if not feature_urls:
        raise ValueError("No feature URLs could be constructed from parameters.")

    feature_dfs = []
    for url in feature_urls:
        try:
            df = fetch_csv(url)
            feature_dfs.append(df)
            print(f"  [OK] Loaded features: {url}")
        except Exception as exc:
            print(f"  [FAIL] Failed to load features {url}: {exc}")

    if not feature_dfs:
        raise FileNotFoundError("Could not load any feature files.")

    features_df = pd.concat(feature_dfs, ignore_index=True)

    # --- Inline target (e.g. blinks) ----------------------------------------
    has_inline_target = hasattr(param.features, "target_column")
    if has_inline_target:
        target_column = param.features.target_column
        if target_column not in features_df.columns:
            # Fall back to 'y'
            if "y" in features_df.columns:
                target_column = "y"
            else:
                raise ValueError(
                    f"Target column '{target_column}' not in features DataFrame."
                )
        X = features_df.drop(columns=[target_column])
        y = features_df[target_column]
    else:
        # --- External targets -----------------------------------------------
        target_url = param.targets.path
        target_df = fetch_csv(target_url)
        print(f"  [OK] Loaded targets: {target_url}")

        # Identify target column
        if "Target" in target_df.columns:
            target_column = "Target"
        elif "target" in target_df.columns:
            target_column = "target"
        elif "y" in target_df.columns:
            target_column = "y"
        else:
            raise ValueError("Cannot find target column (Target/target/y) in targets CSV.")

        # Merge on common column
        common_col = _get_common_join_column(param)
        # Find the actual column name (case-insensitive)
        feat_cols = {c.lower(): c for c in features_df.columns}
        tgt_cols = {c.lower(): c for c in target_df.columns}
        common_feat = feat_cols.get(common_col.lower(), common_col)
        common_tgt = tgt_cols.get(common_col.lower(), common_col)

        if common_feat not in features_df.columns:
            raise ValueError(f"Common column '{common_feat}' not found in features data.")
        if common_tgt not in target_df.columns:
            raise ValueError(f"Common column '{common_tgt}' not found in targets data.")

        merged = features_df.merge(
            target_df[[common_tgt, target_column]],
            left_on=common_feat,
            right_on=common_tgt,
            how="inner",
        )
        if merged.empty:
            raise ValueError("Merge produced 0 rows – check common column values.")

        drop_cols = [common_feat, target_column]
        if common_tgt != common_feat and common_tgt in merged.columns:
            drop_cols.append(common_tgt)
        X = merged.drop(columns=drop_cols, errors="ignore")
        y = merged[target_column]

    # Drop non-numeric columns
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        print(f"  [WARN] Dropping non-numeric columns: {non_numeric}")
        X = X.select_dtypes(include=["number"])

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "lr": "LogisticRegression",
    "logisticregression": "LogisticRegression",
    "rfc": "RandomForest",
    "rbf": "RandomForest",        # alias used by the project
    "randomforest": "RandomForest",
    "svm": "SVM",
    "mlp": "MLP",
    "xgboost": "XGBoost",
}


def _get_model_instance(name: str):
    """Return an sklearn-compatible model instance for *name*."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    canon = MODEL_ALIASES.get(name.lower())
    if canon is None:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(set(MODEL_ALIASES.values()))}")

    if canon == "LogisticRegression":
        return canon, LogisticRegression(max_iter=10000)
    if canon == "SVM":
        return canon, SVC(probability=True)
    if canon == "MLP":
        return canon, MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )
    if canon == "RandomForest":
        return canon, RandomForestClassifier(
            n_estimators=200, criterion="gini", random_state=42
        )
    if canon == "XGBoost":
        from xgboost import XGBClassifier
        return canon, XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    raise ValueError(f"Unhandled model: {canon}")


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train *model*, return metrics dict."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
        roc_curve,
    )

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    start = time.time()
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)
    duration = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)

    # ROC-AUC (needs predict_proba)
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test_imp)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": round(accuracy * 100, 2),
        "roc_auc": round(roc_auc * 100, 2) if roc_auc is not None else None,
        "duration_seconds": round(duration, 2),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling; returns resampled X, y."""
    from imblearn.over_sampling import SMOTE
    from sklearn.impute import SimpleImputer
    import numpy as np

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X_train)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_imp, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), y_res


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_dir: str):
    """Write a summary CSV to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "model_results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "accuracy", "roc_auc", "duration_seconds"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r["model"],
                "accuracy": r["accuracy"],
                "roc_auc": r["roc_auc"],
                "duration_seconds": r["duration_seconds"],
            })
    print(f"\n[FILE] Summary saved to {summary_path}")

    # Also save full JSON report
    json_path = os.path.join(output_dir, "model_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[FILE] Full report saved to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(yaml_path: str) -> list[dict]:
    """
    End-to-end pipeline: load params → fetch data → train models → return results.
    """
    print("=" * 60)
    print("  RealityStream ML Pipeline (Local / Cloud Run)")
    print("=" * 60)

    # 1. Load parameters
    params_path = os.path.abspath(yaml_path)
    if not os.path.exists(params_path):
        print(f"[ERROR] parameters file not found: {params_path}")
        sys.exit(1)

    params = load_parameters(params_path)
    param = DictToObject(OrderedDict(params))
    print(f"\n[PARAMS] Parameters: {params_path}")
    print(f"   Models: {params.get('models', [])}")
    print(f"   Folder: {params.get('folder', 'N/A')}")

    # 2. Fetch data
    print("\n[DATA] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data(param)

    # 3. Determine if SMOTE is needed (class imbalance)
    unique_counts = y_train.value_counts()
    use_smote = False
    if len(unique_counts) == 2:
        ratio = unique_counts.min() / unique_counts.max()
        if ratio < 0.4:
            use_smote = True
            print(f"\n[SMOTE] Class imbalance detected (ratio={ratio:.2f}), applying SMOTE...")
            X_train, y_train = apply_smote(X_train, y_train)
            print(f"   After SMOTE: {len(X_train)} training samples")

    # 4. Train models
    model_names = params.get("models", ["RFC"])
    results = []

    for name in model_names:
        print(f"\n{'-' * 50}")
        try:
            canon_name, model = _get_model_instance(name)
        except ValueError as exc:
            print(f"[WARN] Skipping {name}: {exc}")
            continue

        print(f"[MODEL] Training {canon_name} ({name})...")
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        metrics["model"] = canon_name

        print(f"   Accuracy : {metrics['accuracy']}%")
        if metrics["roc_auc"] is not None:
            print(f"   ROC-AUC  : {metrics['roc_auc']}%")
        print(f"   Time     : {metrics['duration_seconds']}s")
        print(f"\n{metrics['classification_report_text']}")

        results.append(metrics)

    # 5. Save output
    folder_name = params.get("folder", "default")
    output_dir = os.path.join("output", folder_name)
    save_results(results, output_dir)

    print("\n" + "=" * 60)
    print("  [DONE] Pipeline complete!")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RealityStream ML models using a parameters.yaml file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_models.py parameters/parameters.yaml
              python run_models.py parameters/parameters-blinks.yaml
              python run_models.py path/to/custom-params.yaml

            Supported models (specify in YAML 'models' key):
              LR         – Logistic Regression
              RFC / RBF  – Random Forest Classifier
              SVM        – Support Vector Machine
              MLP        – Multi-Layer Perceptron
              XGBoost    – XGBoost Classifier
        """),
    )
    parser.add_argument(
        "yaml",
        help="Path to parameters.yaml (relative or absolute)",
    )
    args = parser.parse_args()
    run_pipeline(args.yaml)


if __name__ == "__main__":
    main()
