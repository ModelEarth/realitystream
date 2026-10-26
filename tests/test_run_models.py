"""Offline self-check for run_models.py. Run: python -m pytest tests  (or python tests/test_run_models.py)."""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
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



# --- Model settings: split, search and hyperparameters (api/model-params.yaml) ---

def test_schema_defaults_match_make_model():
    """Defaults the form shows must be what make_model() builds (searched and only_with fields aside)."""
    hyper = rm.model_params_schema()["hyperparameters"]
    for key, fields in hyper.items():
        current = rm.make_model(key).get_params()
        for name, spec in fields.items():
            value = current.get(name)
            if "only_with" in spec or value is None or value == "deprecated":
                continue
            value = list(value) if isinstance(value, tuple) else value
            assert value == spec["default"], (key, name, value, spec["default"])


def test_search_fields_match_param_grid():
    rng = np.random.default_rng(0)
    for key, fields in rm.model_params_schema()["hyperparameters"].items():
        searched = {name for name, spec in fields.items() if spec.get("search")}
        assert searched == set(rm._param_grid(key, 2, rng) or {}), key


def test_validate_parameters_normalizes():
    settings = rm.validate_parameters({
        "split": {"test_size": 0.3, "random_state": 7.0},
        "search": {"n_iter": 5},
        "hyperparameters": {"xgb": {"learning_rate": "1e-1"}, "mlp": {"hidden_layer_sizes": 64},
                            "lr": {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.2}},
    })
    assert settings["split"] == {"test_size": 0.3, "random_state": 7}
    assert settings["hyperparameters"]["xgboost"] == {"learning_rate": 0.1}
    assert settings["hyperparameters"]["mlp"] == {"hidden_layer_sizes": [64]}
    hashed = rm.validate_parameters({"hyperparameters": {"mlp": {"hidden_layer_sizes": ["50", "50"]}}})  # from a URL hash
    assert hashed["hyperparameters"]["mlp"] == {"hidden_layer_sizes": [50, 50]}
    assert rm.validate_parameters({"models": ["lr"]}) == {"split": {}, "search": {}, "importance": {}, "hyperparameters": {}}


def test_validate_parameters_names_the_field():
    import pytest

    cases = [
        ({"split": {"test_size": 0.9}}, "split.test_size"),
        ({"split": {"shuffle": True}}, "split.shuffle"),
        ({"search": {"n_iter": 2.5}}, "search.n_iter"),
        ({"hyperparameters": {"forest": {}}}, "hyperparameters.forest"),
        ({"hyperparameters": {"rfc": {"criterion": "elasticnet"}}}, "hyperparameters.rfc.criterion"),
        ({"hyperparameters": {"rfc": {"n_jobs": 1000}}}, "hyperparameters.rfc.n_jobs"),
        ({"hyperparameters": {"lr": {"penalty": "l1"}}}, "hyperparameters.lr.penalty"),  # lbfgs can't do l1
        ({"hyperparameters": {"lr": {"l1_ratio": 0.3}}}, "hyperparameters.lr.l1_ratio"),  # needs elasticnet
        ({"hyperparameters": {"svm": {"degree": 4}}}, "hyperparameters.svm.degree"),       # needs poly
        ({"hyperparameters": {"mlp": {"hidden_layer_sizes": [10, 10, 10, 10]}}}, "hyperparameters.mlp.hidden_layer_sizes"),
        ({"hyperparameters": {"rbf": {"feature_importance": 1}}}, "hyperparameters.rbf.feature_importance"),
    ]
    for params, field in cases:
        with pytest.raises(rm.ParameterError) as err:
            rm.validate_parameters(params)
        assert err.value.field == field, (params, err.value.field)


def test_lr_kwargs_for_both_scikit_learn_apis(monkeypatch):
    monkeypatch.setattr(rm, "_sklearn_at_least", lambda major, minor: True)   # 1.8+: no penalty argument
    assert rm._lr_kwargs({"penalty": "l1", "solver": "liblinear"}) == {"solver": "liblinear", "l1_ratio": 1.0}
    assert rm._lr_kwargs({"penalty": "none"}) == {"C": np.inf, "l1_ratio": 0.0}
    assert rm._lr_kwargs({"penalty": "elasticnet", "solver": "saga"}) == {"solver": "saga", "l1_ratio": 0.5}
    monkeypatch.setattr(rm, "_sklearn_at_least", lambda major, minor: False)  # before 1.8
    assert rm._lr_kwargs({"penalty": "l1"}) == {"penalty": "l1"}
    assert rm._lr_kwargs({"penalty": "none"}) == {"penalty": None}
    assert rm._lr_kwargs({"C": 2.0}) == {"C": 2.0}


def test_hyperparameters_reach_the_estimator():
    X, y = _data()
    results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["rfc", "lr"],
                              hyperparameters={"rfc": {"n_estimators": 20, "max_depth": 3},
                                               "lr": {"penalty": "l1", "solver": "liblinear", "C": 0.5}})
    rfc, lr = results
    assert rfc["best_model"].n_estimators == 20 and rfc["best_model"].max_depth == 3
    assert rfc["params_used"] == {"n_estimators": 20, "max_depth": 3, "criterion": "gini", "max_features": "sqrt"}
    assert lr["params_used"]["penalty"] == "l1" and lr["params_used"]["C"] == 0.5
    assert "best_model" not in rm.summarize(results)[0] and "params_used" in rm.summarize(results)[0]


def test_lr_runs_without_penalty_deprecation_warning():
    import warnings

    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["lr"], hyperparameters={"lr": {"penalty": "none"}})
    assert not [w for w in caught if "penalty" in str(w.message)]
    assert results[0]["params_used"]["penalty"] == "none" and "C" not in results[0]["params_used"]


def test_fixed_fields_leave_the_search(monkeypatch):
    import sklearn.model_selection as ms

    real, seen = ms.RandomizedSearchCV, []

    def capture(estimator, param_distributions, **kw):
        seen.append(set(param_distributions))
        return real(estimator, param_distributions, **kw)

    monkeypatch.setattr(ms, "RandomizedSearchCV", capture)
    X, y = _mixed_scale(n=200, minority=50)
    results = rm.train_models(X[:150], y[:150], X[150:], y[150:], ["mlp"], n_iter=2,
                              hyperparameters={"mlp": {"activation": "tanh", "max_iter": 300}})
    assert seen and not {"model__activation", "model__max_iter"} & seen[0]
    assert "model__alpha" in seen[0]
    used = results[0]["params_used"]
    assert used["activation"] == "tanh" and used["max_iter"] == 300


def test_empty_settings_give_unchanged_results():
    X, y = _data()

    def metrics(hyper):
        return [{k: r[k] for k in ("accuracy", "roc_auc", "f1_score")}
                for r in rm.train_models(X[:90], y[:90], X[90:], y[90:], ["rfc", "lr"], hyperparameters=hyper)]

    assert metrics(None) == metrics({}) == metrics({"rfc": {}, "lr": {}})


def test_run_pipeline_reads_split_and_search(tmp_path, monkeypatch):
    X, y = _data()
    seen = {}
    monkeypatch.setattr(rm, "load_data", lambda param, return_groups=False: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    real_split, real_train = rm.split_data, rm.train_models

    def split(X, y, test_size=0.2, random_state=rm.RANDOM_STATE):
        seen["split"] = (test_size, random_state)
        return real_split(X, y, test_size=test_size, random_state=random_state)

    def train(*args, **kw):
        seen["train"] = (kw["random_state"], kw["n_iter"], kw["hyperparameters"])
        return real_train(*args, **kw)

    monkeypatch.setattr(rm, "split_data", split)
    monkeypatch.setattr(rm, "train_models", train)
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [rfc]\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "a"), smote=False)
    assert seen["split"] == (0.2, 42) and seen["train"] == (42, 20, {})   # as before these sections existed
    assert out["settings"] == {"test_size": 0.2, "random_state": 42, "n_iter": 20}
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [rfc]\nsplit:\n  test_size: 0.3\n  random_state: 5\n"
                      "search:\n  n_iter: 4\nhyperparameters:\n  rfc: {n_estimators: 30}\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "b"), smote=False)
    assert seen["split"] == (0.3, 5) and seen["train"] == (5, 4, {"rfc": {"n_estimators": 30}})
    assert out["no_smote"][0]["params_used"]["n_estimators"] == 30


def test_run_pipeline_rejects_bad_settings_before_loading_data(tmp_path, monkeypatch):
    import pytest

    def no_load(*a, **kw):
        raise AssertionError("data loaded before the settings were checked")

    monkeypatch.setattr(rm, "load_data", no_load)
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [rfc]\nhyperparameters:\n  rfc: {max_depth: 0}\n",
                      encoding="utf-8")
    with pytest.raises(rm.ParameterError) as err:
        rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"))
    assert err.value.field == "hyperparameters.rfc.max_depth"

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


def test_empty_models_still_write_the_report(tmp_path, monkeypatch):
    X, y = _mixed_scale(n=200, minority=50, big=5.0)
    monkeypatch.setattr(rm, "load_data", lambda param: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: []\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"), smote=None)
    assert out["no_smote"] == [] and out["smote"] == [] and "baseline" in out
    assert (tmp_path / "r" / "parameters.yaml").exists()
    # files.json lists the folder's files for its index.html
    import json as _json
    listed = _json.loads((tmp_path / "r" / "files.json").read_text(encoding="utf-8"))["files"]
    assert "parameters.yaml" in listed and "files.json" not in listed and "index.html" not in listed
    # A YAML without a models line still runs rfc
    params.write_text("folder: t\nfeatures:\n  path: x\n", encoding="utf-8")
    assert rm.load_parameters(str(params))["models"] == ["rfc"]
    # Text forms in the presets: comma-separated keys, and all
    params.write_text("models: rbf,rfc, XGBoost\n", encoding="utf-8")
    assert rm.load_parameters(str(params))["models"] == ["rbf", "rfc", "xgboost"]
    params.write_text("models: all\n", encoding="utf-8")
    assert rm.load_parameters(str(params))["models"] == ["lr", "rfc", "rbf", "svm", "mlp", "xgboost"]


def test_run_pipeline_writes_charts_and_lists_failed_models(tmp_path, monkeypatch):
    """Every trained model reaches the PNG charts; one that fails is named, with its reason, in README.md."""
    X, y = _mixed_scale(n=200, minority=50, big=5.0)
    monkeypatch.setattr(rm, "load_data", lambda param: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    monkeypatch.setattr(rm.platform, "system", lambda: "Darwin")  # rbf fails off Linux
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr, rfc, rbf, svm, mlp, xgboost]\n"
                      "search: {n_iter: 2}\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"), smote=True)
    assert list(out["errors"]) == ["rbf"] and "Linux" in out["errors"]["rbf"]
    assert "y_pred" not in out["smote"][0] and "predictor" not in out["smote"][0]
    import json as _json
    listed = _json.loads((tmp_path / "r" / "files.json").read_text(encoding="utf-8"))["files"]
    for name in ["metrics_dashboard.png", "metrics_comparison.png", "roc_curves_comparison.png", "confusion_matrices.png",
                 "training_time_comparison.png"] + [f"feature_importance_{k}.png" for k in ("lr", "rfc", "svm", "mlp", "xgboost")]:
        assert name in listed and (tmp_path / "r" / name).read_bytes()[:4] == b"\x89PNG"
    readme = (tmp_path / "r" / "README.md").read_text(encoding="utf-8")
    assert "Random Bits Forest: The Random Bits Forest binary runs on Linux only" in readme
    assert "charts show the run with SMOTE" in readme


def test_data_warnings_and_load_failures_reach_the_readme(tmp_path, monkeypatch):
    """One positive county among 16 rows: the README says so. A load error still leaves a README naming it."""
    X, y = _data(n=16)
    y = pd.Series([0] * 15 + [1])
    monkeypatch.setattr(rm, "load_data", lambda param: (X, y))
    monkeypatch.setattr(rm, "setup_report_folder", lambda d: os.makedirs(d, exist_ok=True))
    params = tmp_path / "p.yaml"
    params.write_text("folder: t\nfeatures:\n  path: x\nmodels: [lr, rfc]\n", encoding="utf-8")
    out = rm.run_pipeline(str(params), report_dir=str(tmp_path / "r"), smote=False)
    readme = (tmp_path / "r" / "README.md").read_text(encoding="utf-8")
    assert "Data warnings:" in readme and "- Only 1 of 16 rows have target 1" in readme
    assert any(w.startswith("Only 1 of 16 rows have target 1") for w in out["warnings"])

    def missing(param):
        raise ValueError("Merge produced 0 rows. Check the common column values.")
    monkeypatch.setattr(rm, "load_data", missing)
    with pytest.raises(ValueError):
        rm.run_pipeline(str(params), report_dir=str(tmp_path / "r2"))
    assert "stopped before training: ValueError: Merge produced 0 rows" in (tmp_path / "r2" / "README.md").read_text(
        encoding="utf-8")


def test_rbf_runs_once_per_pass_and_reaches_the_charts(tmp_path, monkeypatch):
    """A stand-in for the Linux RBF binary: scores rows by their first column, and counts its runs."""
    calls = tmp_path / "calls"
    fake = tmp_path / "rbf"
    fake.write_text(f"#!{sys.executable}\nimport sys\nopen({str(calls)!r}, 'a').write('run\\n')\n"
                    "testx, out = sys.argv[-2:]\n"
                    "rows = [float(line.split(',')[0]) for line in open(testx)]\n"
                    "open(out, 'w').write('\\n'.join(str(min(1, max(0, v / 4 + 0.5))) for v in rows))\n",
                    encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(rm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rm.RandomBitsForest, "_ensure_binary", lambda self: str(fake))
    X, y = _data()
    results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["rbf"])
    assert [r["model_type"] for r in results] == ["rbf"] and results[0]["roc_auc"] is not None
    assert calls.read_text().count("run") == 1  # predict() reused predict_proba()'s run
    # Feature importance is optional for rbf, since it trains it once more
    assert "rbf" not in rm.feature_importances(results, list(X.columns), X_test=X[90:], y_test=y[90:])
    assert calls.read_text().count("run") == 1
    hyper = rm.validate_parameters({"hyperparameters": {"rbf": {"feature_importance": "true"}}})["hyperparameters"]
    results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["rbf"], hyperparameters=hyper)
    assert results[0]["params_used"]["feature_importance"] is True
    # Permutation importance stacks every shuffled copy into one more run, and finds the column it scores by
    importances = rm.feature_importances(results, list(X.columns), X_test=X[90:], y_test=y[90:])
    assert calls.read_text().count("run") == 3
    assert importances["rbf"].attrs["method"] == "permutation" and importances["rbf"]["Feature"][0].startswith(X.columns[0])
    from charts import write_charts
    written = write_charts(str(tmp_path), results, y[90:], importances)
    assert written == ["metrics_dashboard.png", "metrics_comparison.png", "roc_curves_comparison.png",
                       "confusion_matrices.png", "training_time_comparison.png", "feature_importance_rbf.png"]


def test_upload_reports_sends_images_as_base64_blobs(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Report", encoding="utf-8")
    png = b"\x89PNG\r\n\x1a\n\x00\xff"
    (tmp_path / "chart.png").write_bytes(png)
    sent = []

    class Resp:
        def __init__(self, body):
            self.body = body
        def raise_for_status(self):
            pass
        def json(self):
            return self.body

    def request(method, url, json=None, **kw):
        sent.append((method, url, json))
        return Resp({"object": {"sha": "head"}, "tree": {"sha": "base"}, "sha": "new"})

    monkeypatch.setattr(rm.requests, "request", request)
    rm.upload_reports(str(tmp_path), repo="o/r", year="2026", subfolder="run-x", token="t")
    import base64 as _b64
    blob = next(body for method, url, body in sent if url.endswith("/git/blobs"))
    assert blob["encoding"] == "base64" and _b64.b64decode(blob["content"]) == png
    tree = next(body for method, url, body in sent if url.endswith("/git/trees"))["tree"]
    assert {"path": "2026/run-x/chart.png", "mode": "100644", "type": "blob", "sha": "new"} in tree
    assert {"path": "2026/run-x/README.md", "mode": "100644", "type": "blob", "content": "# Report"} in tree


def test_push_report_auto_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "get_env", lambda name, default=None: default)
    assert rm.push_report(str(tmp_path), "auto")[1]["token"] == "missing"
    with pytest.raises(rm.MissingKey):
        rm.push_report(str(tmp_path), True)
    assert rm.push_report(str(tmp_path), False) == (None, {"token": None, "pushed_to": None, "message": ""})
    assert re.fullmatch(r"\d{4}/run-\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d", rm.report_name())


def test_check_reports_token(monkeypatch):
    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self.body = code, body or {}
        def json(self):
            return self.body
    assert rm.check_reports_token("")["token"] == "missing"
    for code, want in [(401, "invalid"), (422, "valid"), (403, "no_access"), (404, "no_access"), (500, "unknown")]:
        monkeypatch.setattr(rm.requests, "post", lambda *a, _r=Resp(code), **k: _r)
        assert rm.check_reports_token("t")["token"] == want
    monkeypatch.setattr(rm.requests, "post", lambda *a, **k: Resp(403))
    assert "Read and write" in rm.check_reports_token("t")["message"]
    monkeypatch.setattr(rm.requests, "post", lambda *a, **k: Resp(422))
    assert "(from this browser)" in rm.check_reports_token("t", "GITHUB_REPORTS_TOKEN (from this browser)")["message"]


def test_reports_repo_and_site(monkeypatch):
    settings = {}
    monkeypatch.setattr(rm, "get_env", lambda name, default=None: settings.get(name, default))
    assert rm.reports_repo() == "modelearth/reports"
    assert rm.reports_site() == "https://model.earth/reports/"
    settings["GITHUB_REPORTS_REPO"] = "Someone/reports"
    assert rm.reports_site() == "https://someone.github.io/reports/"
    assert rm.reports_site("someone/someone.github.io") == "https://someone.github.io/"
    settings["GITHUB_REPORTS_SITE"] = "https://reports.example.org"
    assert rm.reports_site() == "https://reports.example.org/"


def test_push_report_view_url(monkeypatch, tmp_path):
    settings = {"GITHUB_REPORTS_TOKEN": "t"}
    monkeypatch.setattr(rm, "get_env", lambda name, default=None: settings.get(name, default))
    monkeypatch.setattr(rm, "upload_reports", lambda d, repo, year, subfolder, token: f"{year}/{subfolder}")
    pushed, github = rm.push_report(str(tmp_path), "auto", name="2026/run-x")
    assert pushed == "2026/run-x" and github["view_url"] == "https://model.earth/reports/2026/run-x/"
    assert github["repo"] == "modelearth/reports"
    assert github["message"].index("View at https://model.earth/reports/2026/run-x/") < github["message"].index("Pushed to")


def test_wide_data_importance_options(monkeypatch):
    """Over PERMUTATION_FEATURES, svm's permutation chart is skipped unless prescreen or time_budget is set."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 250)), columns=[f"f{i}" for i in range(250)])
    y = pd.Series((X["f7"] + rng.normal(scale=0.3, size=120) > 0).astype(int))
    results = rm.train_models(X[:90], y[:90], X[90:], y[90:], ["svm"])

    def run(options):
        notes = {}
        imp = rm.feature_importances(results, list(X.columns), map_naics=False, X_test=X[90:], y_test=y[90:],
                                     X_train=X[:90], y_train=y[:90], options=options, notes=notes)
        return imp, notes.get("svm", "")

    imp, note = run(None)
    assert "svm" not in imp and "Not drawn: 250 features" in note
    imp, note = run({"prescreen": True})
    assert len(imp["svm"]) == 200 and imp["svm"]["Feature"][0] == "f7" and "Shuffled the 200 of 250" in note
    imp, note = run({"time_budget": True, "budget_seconds": 600})
    assert len(imp["svm"]) == 250 and note.startswith("All 250 features shuffled")
    # Over the budget: skipped, or prescreened when that's checked too
    monkeypatch.setattr(rm, "PERMUTATION_BATCH", 8)
    imp, note = run({"time_budget": True, "budget_seconds": 1e-9})
    assert "svm" not in imp and "over the 1e-09 s budget" in note and "check prescreen)" in note
    imp, note = run({"time_budget": True, "budget_seconds": 1e-9, "prescreen": True})
    assert len(imp["svm"]) == 200 and "over the" in note and "Shuffled the 200" in note
    readme = rm.report_readme({"folder": "t", "models": ["svm"]}, results, [], importance={"svm": note})
    assert "Feature importance:" in readme and "Support Vector Machine: Shuffling all 250" in readme


def test_importance_settings_validate():
    import pytest

    checked = rm.validate_parameters({"importance": {"prescreen": "true", "time_budget": True, "budget_seconds": 30}})
    assert checked["importance"] == {"prescreen": True, "time_budget": True, "budget_seconds": 30}
    with pytest.raises(rm.ParameterError) as err:
        rm.validate_parameters({"importance": {"budget_seconds": 30}})
    assert err.value.field == "importance.budget_seconds" and "time_budget is true (time_budget is false)" in str(err.value)
