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
    monkeypatch.setattr(main, "ON_CLOUD_RUN", True)  # temporary report folders, as on Cloud Run
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
    # With the passphrase, the report is pushed to GitHub when the token is set ("auto")
    assert calls == [{"report_dir": calls[0]["report_dir"], "upload": "auto", "smote": False, "name": calls[0]["name"],
                      "token": None, "token_label": "GITHUB_REPORTS_TOKEN"}]
    assert _runs() == 1


def test_local_runs_save_reports_in_the_webroot(monkeypatch, tmp_path):
    client, calls = _client(monkeypatch)
    monkeypatch.setattr(main, "ON_CLOUD_RUN", False)
    monkeypatch.setattr(main, "WEBROOT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "REPORTS_DIR", str(tmp_path / "reports"))
    (tmp_path / "reports").mkdir()
    data = client.post("/run", data=YAML).get_json()
    name = calls[0]["name"]
    assert calls[0]["report_dir"] == os.path.join(str(tmp_path), "reports", *name.split("/"))
    assert data["output_path"] == f"reports/{name}/" and data["report_name"] == name
    # Sent without the Team Passphrase, so nothing is pushed
    assert calls[0]["upload"] == False
    assert "Team Passphrase" in data["github"]["message"]


def test_default_settings_without_passphrase_still_run(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run", data=YAML)
    assert resp.status_code == 200 and len(calls) == 1


def test_github_token_check_is_cached(monkeypatch):
    client, _ = _client(monkeypatch)
    main._github_checks.clear()
    asked = []
    monkeypatch.setattr(rm, "check_reports_token", lambda token, label: asked.append(label) or {"token": "invalid", "message": "m"})
    assert client.get("/github-token").get_json()["token"] == "invalid"
    assert client.get("/github-token").get_json()["source"] == "server"
    assert len(asked) == 1
    # A token saved in the browser is checked instead, and named as the browser's
    data = client.get("/github-token", headers={"X-GitHub-Token": "browser-token"}).get_json()
    assert data["source"] == "browser" and asked[-1] == main.BROWSER_TOKEN_LABEL and len(asked) == 2


def test_browser_token_pushes_keyless_runs(monkeypatch):
    client, calls = _client(monkeypatch)
    resp = client.post("/run", data=YAML, headers={"X-GitHub-Token": "browser-token"})
    assert resp.status_code == 200
    assert calls[0]["upload"] == "auto" and calls[0]["token"] == "browser-token"
    assert calls[0]["token_label"] == main.BROWSER_TOKEN_LABEL
    assert "X-GitHub-Token" in client.options("/run", headers={"Origin": "http://localhost:8887"}).headers[
        "Access-Control-Allow-Headers"]


def test_streamed_run_sends_the_run_id_first_and_keeps_the_result(monkeypatch):
    import json
    import threading

    client, _ = _client(monkeypatch)
    monkeypatch.setattr(main, "KEEPALIVE_SECONDS", 0.05)
    monkeypatch.setattr(main, "POLL_WAIT_SECONDS", 0.05)
    release = threading.Event()
    monkeypatch.setattr(rm, "run_pipeline", lambda path, **kw: release.wait(5) and {"folder": "t"})
    resp = client.post("/run?stream=1", data=YAML, headers={"X-API-Key": PASSPHRASE}, buffered=False)
    assert resp.status_code == 200 and resp.mimetype == "application/x-ndjson"
    stream = resp.response
    first = json.loads(next(stream))
    assert first["status"] == "running" and first["report_name"].startswith("20")
    run_id = first["run_id"]
    # While it runs, GET /run/<run_id> answers 202, and the stream sends keep-alive lines
    assert client.get(f"/run/{run_id}").status_code == 202
    assert json.loads(next(stream)) == {"status": "running", "seconds": 0}
    release.set()
    lines = [json.loads(line) for line in stream if line.strip()]
    last = lines[-1]
    assert last["status"] == "success" and last["http_status"] == 200 and last["run_id"] == run_id
    # The result stays available by ID, for a page whose connection dropped
    later = client.get(f"/run/{run_id}")
    assert later.status_code == 200 and later.get_json()["folder"] == "t"
    assert client.get("/run/not-a-run").status_code == 404


def test_unstreamed_run_and_errors_keep_their_status(monkeypatch):
    client, _ = _client(monkeypatch)

    def missing(path, **kw):
        raise rm.MissingKey("GITHUB_REPORTS_TOKEN")
    monkeypatch.setattr(rm, "run_pipeline", missing)
    resp = client.post("/run", data=YAML, headers={"X-API-Key": PASSPHRASE})
    assert resp.status_code == 400 and resp.get_json()["status"] == "missing_key" and resp.get_json()["run_id"]
    import json
    lines = client.post("/run?stream=1", data=YAML, headers={"X-API-Key": PASSPHRASE}).get_data(as_text=True)
    last = json.loads(lines.strip().splitlines()[-1])
    assert last["status"] == "missing_key" and last["http_status"] == 400
