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


def _mixed_scale(n=400, minority=200, seed=1, big=1000.0):
    """Binary data whose informative feature is on a large scale, so unscaled lr/svm/mlp struggle."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n - minority) + [1] * minority)
    signal = rng.normal(loc=np.where(y == 1, 2.0, 0.0), scale=1.0) * big
    X = pd.DataFrame({"big": signal, "small": rng.normal(size=n)})
    idx = rng.permutation(n)
    return X.iloc[idx].reset_index(drop=True), pd.Series(y[idx]).reset_index(drop=True)


def test_split_data_is_stratified():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200)})
    y = pd.Series([1] * 40 + [0] * 160)   # 20% positive
    X_train, X_test, y_train, y_test = rm.split_data(X, y)
    assert set(np.unique(y_train)) == {0, 1} and set(np.unique(y_test)) == {0, 1}
    assert abs(np.mean(np.asarray(y_test) == 1) - 0.2) < 0.05


def test_single_class_test_set_gives_none_roc_auc():
    X, y = _data(n=80)
    y_test_single = pd.Series([0] * len(X[60:]))
    results = rm.train_models(X[:60], y[:60], X[60:], y_test_single, ["lr"])
    assert results[0]["roc_auc"] is None


def test_scale_sensitive_models_reach_high_roc():
    X, y = _mixed_scale()
    X_train, X_test, y_train, y_test = rm.split_data(X, y)
    results = rm.train_models(X_train, y_train, X_test, y_test, ["lr", "svm", "mlp"], n_iter=3)
    for r in results:
        assert r["roc_auc"] is not None and r["roc_auc"] > 0.85, (r["model_type"], r["roc_auc"])


def test_smote_runs_per_fold_inside_search(monkeypatch):
    import imblearn.over_sampling as ios
    import sklearn.model_selection as ms

    seen = []
    real_smote = ios.SMOTE

    class SpySMOTE(real_smote):
        def fit_resample(self, X, y=None, **kw):
            seen.append(len(X))
            return super().fit_resample(X, y, **kw)

    real_search = ms.RandomizedSearchCV

    def serial_search(*a, **k):
        k["n_jobs"] = 1   # run in-process so the SMOTE spy sees every fold's calls
        return real_search(*a, **k)

    monkeypatch.setattr(ios, "SMOTE", SpySMOTE)
    monkeypatch.setattr(ms, "RandomizedSearchCV", serial_search)

    rng = np.random.default_rng(2)
    n = 300
    y = np.array([0] * 240 + [1] * 60)
    X = pd.DataFrame({"a": rng.normal(loc=np.where(y == 1, 1.0, 0.0), size=n), "b": rng.normal(size=n)})
    idx = rng.permutation(n)
    X, y = X.iloc[idx].reset_index(drop=True), pd.Series(y[idx])
    X_train, X_test, y_train, y_test = rm.split_data(X, y)
    results = rm.train_models(X_train, y_train, X_test, y_test, ["xgboost"], n_iter=2, smote=True)
    assert seen, "SMOTE was never called"
    assert min(seen) < len(X_train)      # per-fold training parts are smaller than the full split
    report = results[0]["classification_report"]
    assert report["0"]["support"] + report["1"]["support"] == len(X_test)   # test support == test size


def test_smote_skips_when_a_class_has_fewer_than_two_rows():
    X, _ = _data(n=50)
    y = pd.Series([0] * 49 + [1] * 1)
    assert rm.train_models(X, y, X, y, ["lr"], smote=True) == []


def test_baseline_lift_and_csv_header(tmp_path, monkeypatch):
    import csv as _csv
    X, y = _mixed_scale(n=200, minority=50, big=5.0)
    monkeypatch.setattr(rm, "load_data", lambda param: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr]\n", encoding="utf-8")
    report = tmp_path / "report"
    out = rm.run_pipeline(str(params), report_dir=str(report), smote=False)
    assert "baseline" in out and "balanced_accuracy" in out["baseline"]
    assert out["no_smote"] and all("lift_over_baseline" in r for r in out["no_smote"])
    with open(report / "model_performance_report_no_smote.csv", encoding="utf-8") as fh:
        assert next(_csv.reader(fh)) == rm.REPORT_COLUMNS


def _grouped(n_per_group=40, n_groups=6, seed=0):
    """Grouped data: each group (state) carries its own signal shift, so the group matters."""
    rng = np.random.default_rng(seed)
    frames, ys, gs = [], [], []
    for g in range(n_groups):
        base = rng.normal(loc=g * 0.4, scale=1.0, size=n_per_group)
        frames.append(pd.DataFrame({"f1": base, "f2": rng.normal(size=n_per_group)}))
        ys.append((base + rng.normal(scale=0.5, size=n_per_group) > g * 0.4).astype(int))
        gs.append(np.full(n_per_group, g))
    return (pd.concat(frames, ignore_index=True), pd.Series(np.concatenate(ys)), pd.Series(np.concatenate(gs)))


def test_load_data_return_groups_is_state_fips(monkeypatch):
    features = pd.DataFrame({"Fips": ["1001", "1003", "6001", "6003"], "Emp-1": [1.0, 2.0, 3.0, 4.0]})
    targets = pd.DataFrame({"Fips": ["1001", "1003", "6001", "6003"], "Target": [0, 1, 0, 1]})
    param = rm.DictToObject({"features": {"path": "https://x/data.csv", "common": "Fips"},
                             "targets": {"path": "https://x/t.csv", "common": "Fips"}})
    monkeypatch.setattr(rm, "fetch_csv", lambda url: features if "data" in url else targets)
    X, y, groups = rm.load_data(param, return_groups=True)
    assert [int(g) for g in groups] == [1, 1, 6, 6]           # county FIPS // 1000 == state
    assert len(rm.load_data(param)) == 2                      # default call still returns (X, y)


def test_cv_fold_count_and_default_stratified():
    X, y, groups = _grouped()
    calls = []
    import run_models as _rm
    orig = _rm._cv_fold_scores
    try:
        _rm._cv_fold_scores = lambda *a: (calls.append(1) or
            {"roc_auc": 0.5, "pr_auc": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5})
        results, cv_used = _rm.cross_validate_models(X, y, groups, ["lr"], folds=4, cv="stratified")
    finally:
        _rm._cv_fold_scores = orig
    assert cv_used == "stratified"
    assert len(calls) == (1 + 1) * 4                          # (baseline + 1 model) x folds
    assert results[0]["model"] == "MajorityBaseline"


def test_cv_group_folds_hold_whole_states_out():
    X, y, groups = _grouped(n_groups=6)
    seen = []
    import run_models as _rm
    orig = _rm._cv_fold_scores

    def spy(estimator, X_tr, y_tr, X_te, y_te):
        seen.append((set(groups.iloc[list(X_tr.index)]), set(groups.iloc[list(X_te.index)])))
        return {"roc_auc": 0.5, "pr_auc": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5}

    try:
        _rm._cv_fold_scores = spy
        _, cv_used = _rm.cross_validate_models(X, y, groups, ["lr"], folds=3, cv="group")
    finally:
        _rm._cv_fold_scores = orig
    assert cv_used == "group" and seen
    for train_g, test_g in seen:
        assert train_g.isdisjoint(test_g)                    # no state in both train and test


def test_cv_group_falls_back_to_stratified():
    X, y, groups = _grouped(n_per_group=60, n_groups=3)      # 3 groups < 5 folds
    import run_models as _rm
    orig = _rm._cv_fold_scores
    try:
        _rm._cv_fold_scores = lambda *a: {"roc_auc": 0.5, "pr_auc": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5}
        _, cv_used = _rm.cross_validate_models(X, y, groups, ["lr"], folds=5, cv="group")
    finally:
        _rm._cv_fold_scores = orig
    assert cv_used == "stratified"


def test_run_pipeline_absent_evaluation_matches_a1(tmp_path, monkeypatch):
    X, y = _mixed_scale(n=200, minority=50, big=5.0)
    monkeypatch.setattr(rm, "load_data", lambda param, return_groups=False: (X, y, None) if return_groups else (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr]\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"), smote=False)
    assert "cv" not in out
    assert not (tmp_path / "r" / "model_performance_cv.csv").exists()


def test_run_pipeline_with_evaluation_writes_cv(tmp_path, monkeypatch):
    X, y, groups = _grouped(n_per_group=50, n_groups=6)
    monkeypatch.setattr(rm, "load_data",
                        lambda param, return_groups=False: (X, y, groups) if return_groups else (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr]\nevaluation:\n  cv: stratified\n  folds: 5\n",
                      encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"), smote=False)
    assert out["cv"]["mode"] == "stratified" and out["cv"]["folds"] == 5
    import csv as _csv
    with open(tmp_path / "r" / "model_performance_cv.csv", encoding="utf-8") as fh:
        assert next(_csv.reader(fh)) == rm.CV_REPORT_COLUMNS


def test_cli_writes_run_summary(tmp_path, monkeypatch):
    """main() must write run_summary.json end to end (regression for the missing `import json`)."""
    X, y = _data(n=60)
    monkeypatch.setattr(rm, "load_data", lambda param: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "params.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr]\n", encoding="utf-8")
    report = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["run_models.py", str(params), "--report-dir", str(report)])
    rm.main()
    assert (report / "run_summary.json").exists()


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
