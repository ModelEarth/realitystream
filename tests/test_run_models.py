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


def test_join_aliases_fips_padding_and_index_column(monkeypatch):
    # all-years style yaml: features.id / targets.column / join.key, unpadded FIPS vs padded, index column
    param = rm.DictToObject({"features": {"id": "Fips", "path": "x"}, "targets": {"path": "y", "column": "Target"},
                             "join": {"key": "Fips"}})
    assert rm._common_column(param) == "Fips"
    feats = pd.DataFrame({"Unnamed: 0": [0, 1], "Fips": [1001, 1003], "Emp-11": [5.0, 7.0]})
    targs = pd.DataFrame({"Fips": ["01001", "01003"], "Target": [1, 0]})
    monkeypatch.setattr(rm, "fetch_csv", lambda url: feats.copy() if url == "x" else targs.copy())
    X, y = rm.load_data(param)
    assert list(X.columns) == ["Emp-11"] and list(y) == [1, 0]


def test_naics_name_keeps_prefix_and_year(monkeypatch):
    monkeypatch.setitem(rm._NAICS_NAMES, 2, {"11": "Agriculture"})
    monkeypatch.setitem(rm._NAICS_NAMES, 6, {"454310": "Fuel Dealers"})
    assert rm.naics_name("Emp-11") == "Emp-11-Agriculture"
    assert rm.naics_name("Pay-454310-2019") == "Pay-454310-Fuel Dealers-2019"
    assert rm.naics_name("Km2") == "Km2"


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
    print("(alias and naics_name checks need pytest's monkeypatch)")
    print("all checks passed")
