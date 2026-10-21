"""Offline checks for the Flask API in api/main.py. Run: python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
import main  # noqa: E402
import run_models as rm  # noqa: E402

PASSPHRASE = "test-passphrase"
YAML = "folder: t\nfeatures:\n  path: x\nmodels: [rfc]\n"


def _client(monkeypatch):
    """Test client with a Team Passphrase configured, and a pipeline that returns at once."""
    real_get_env = rm.get_env
    monkeypatch.setattr(rm, "get_env", lambda name, default=None:
                        PASSPHRASE if name == "REALITYSTREAM_API_KEY" else real_get_env(name, default))
    calls = []
    monkeypatch.setattr(rm, "run_pipeline", lambda path, **kw: calls.append(kw) or {"folder": "t"})
    monkeypatch.setattr(main, "usage", main.UsageStore())  # fresh in-memory daily totals
    return main.app.test_client(), calls


def _runs(caller="127.0.0.1"):
    return main.usage.caller_runs(caller) + main.usage.caller_runs("free:" + caller)


def test_model_params_serves_the_schema(monkeypatch):
    client, _ = _client(monkeypatch)
    data = client.get("/model-params").get_json()
    assert data["hyperparameters"]["rfc"]["n_estimators"]["default"] == 100
    assert data["split"]["test_size"]["max"] == 0.5


def test_settings_without_passphrase_need_the_key(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run", data=YAML + "hyperparameters:\n  rfc: {n_estimators: 50}\n")
    assert resp.status_code == 403
    assert resp.get_json()["status"] == "key_needed" and resp.get_json()["sections"] == ["hyperparameters"]
    assert not calls and _runs() == 0


def test_invalid_setting_is_400_and_not_counted(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run", data=YAML + "split:\n  test_size: 0.9\n", headers={"X-API-Key": PASSPHRASE})
    assert resp.status_code == 400 and resp.get_json()["field"] == "split.test_size"
    assert not calls and _runs() == 0


def test_settings_with_passphrase_run(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run?smote=0", data=YAML + "hyperparameters:\n  rfc: {n_estimators: 50}\n",
                       headers={"X-API-Key": PASSPHRASE})
    assert resp.status_code == 200 and resp.get_json()["status"] == "success"
    assert calls == [{"report_dir": calls[0]["report_dir"], "upload": False, "smote": False}]
    assert _runs() == 1


def test_default_settings_without_passphrase_still_run(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run", data=YAML)
    assert resp.status_code == 200 and len(calls) == 1
