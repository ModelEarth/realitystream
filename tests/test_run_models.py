"""Offline self-check for run_models.py. Run: python -m pytest tests  (or python tests/test_run_models.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_models as rm  # noqa: E402


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"Emp-454310": rng.normal(size=n), "Pay-11": rng.normal(size=n), "empty": np.nan})
    y = pd.Series((X["Emp-454310"] + rng.normal(scale=0.5, size=n) > 0.8).astype(int))
    return X, y


def test_state_all_expands_to_50_states():
    param = rm.DictToObject({"features": {"path": "https://x/{state}-{naics}-{year}.csv", "state": "all",
                                          "naics": [2], "startyear": 2021, "endyear": 2021}})
    urls = rm.build_feature_urls(param)
    assert len(urls) == 50 and "https://x/ME-2-2021.csv" in urls


def test_smote_keeps_all_columns_and_balances():
    X, y = _data()
    Xr, yr = rm.apply_smote(X, y)
    assert list(Xr.columns) == list(X.columns)
    assert yr.value_counts().nunique() == 1


def test_train_models_and_report_columns(tmp_path):
    X, y = _data()
    results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["rfc", "lr", "xgboost"], n_iter=2)
    assert [r["model_type"] for r in results] == ["rfc", "lr", "xgboost"]
    table = rm.results_table(results)
    assert list(table.columns) == rm.REPORT_COLUMNS
    imp = rm.feature_importances(results, list(X.columns))
    assert set(imp) == {"rfc", "lr", "xgboost"}
    assert list(imp["xgboost"].columns) == ["Feature", "Importance"]


def test_missing_key_has_guidance(monkeypatch):
    monkeypatch.delenv("DATACOMMONS_API_KEY", raising=False)
    monkeypatch.setattr(rm, "env_file_path", lambda: None)
    try:
        rm.require_env("DATACOMMONS_API_KEY")
    except rm.MissingKey as exc:
        assert "datacommons" in exc.how_to_get_it.lower()
    else:
        raise AssertionError("expected MissingKey")


if __name__ == "__main__":
    import tempfile

    class _Patch:
        def delenv(self, k, raising=False): os.environ.pop(k, None)
        def setattr(self, obj, name, value): setattr(obj, name, value)

    test_state_all_expands_to_50_states()
    test_smote_keeps_all_columns_and_balances()
    test_train_models_and_report_columns(tempfile.mkdtemp())
    test_missing_key_has_guidance(_Patch())
    print("all checks passed")
