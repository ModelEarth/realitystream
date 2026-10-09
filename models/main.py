"""
RealityStream serverless API for Google Cloud Run: Flask wrapper around run_models.run_pipeline.

    GET  /                     list of endpoints
    GET  /health               liveness probe
    GET  /parameters           preset YAML files in parameters/
    POST /run[?upload=1]       body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}
                               returns the run summary as JSON; upload=1 pushes the report folder

A missing key (for example GITHUB_REPORTS_TOKEN when upload=1) returns HTTP 400 with
`how_to_get_it`, which the front end shows to the user instead of a stack trace.
Each caller gets DAILY_RUNS_PER_USER runs a day (default 20); past that, HTTP 429.
Runs also stop for the day once their estimated cost reaches DAILY_COST_LIMIT_USD (default $0.20).
When REALITYSTREAM_API_KEY is set, /parameters and /run require header X-API-Key.

Settings come from run_models.get_env: environment, then the OS credential store, then the
env file named in automation/paths.yaml. On Cloud Run, set them with --set-env-vars or --set-secrets.

Deploy (no Dockerfile, Google Cloud buildpacks read requirements.txt and Procfile):
    cd realitystream && ./deploy-cloud-run.sh
Local (from the realitystream folder):
    python models/main.py
"""

import base64
import hmac
import json
import os
import sys
import tempfile
import time
from datetime import date

from flask import Flask, jsonify, request

REALITYSTREAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETERS_DIR = os.path.join(REALITYSTREAM_DIR, "parameters")

# run_models.py resides in the realitystream folder
if REALITYSTREAM_DIR not in sys.path:
    sys.path.insert(0, REALITYSTREAM_DIR)

app = Flask(__name__)

DAILY_RUNS_PER_USER = int(os.environ.get("DAILY_RUNS_PER_USER", "20"))
_runs = {}  # (day, caller) -> count. Per-instance memory, exact with --max-instances 1; use Firestore if that changes.

# Daily spending cap. Run time is priced at Cloud Run request-based rates for 4 vCPU + 2 GiB
# (4 x $0.000024 per vCPU-second + 2 x $0.0000025 per GiB-second), ignoring the free tier.
DAILY_COST_LIMIT_USD = float(os.environ.get("DAILY_COST_LIMIT_USD", "0.20"))
COST_PER_SECOND_USD = float(os.environ.get("COST_PER_SECOND_USD", "0.000101"))
_spent = {}  # day -> estimated USD. Same per-instance caveat as _runs.

ALLOWED_ORIGINS = {"http://localhost:8887", "http://127.0.0.1:8887"}


@app.after_request
def add_cors_headers(response):
    """Allow the local webroot (port 8887) to call this API."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def authorized():
    """True when no REALITYSTREAM_API_KEY is configured, or when X-API-Key matches it."""
    from run_models import get_env

    expected = get_env("REALITYSTREAM_API_KEY", "")
    if not expected:
        return True
    return hmac.compare_digest(request.headers.get("X-API-Key", ""), expected)


def caller_id():
    """Who is calling: the email in the Cloud Run bearer token (already verified by Cloud Run), else the client IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth.count(".") == 2:
        try:
            payload = auth.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            return claims.get("email") or claims.get("sub") or "token"
        except Exception:
            pass
    return request.headers.get("X-Forwarded-For", request.remote_addr or "anon").split(",")[0].strip()


def throttled(caller):
    """True when the caller has used today's run allowance; otherwise counts this run."""
    key = (date.today().isoformat(), caller)
    if _runs.get(key, 0) >= DAILY_RUNS_PER_USER:
        return True
    _runs[key] = _runs.get(key, 0) + 1
    return False


def preset_files():
    return sorted(f for f in os.listdir(PARAMETERS_DIR) if f.endswith((".yaml", ".yml")))


@app.route("/", methods=["GET"])
def index():
    """List the endpoints, so opening the service URL in a browser shows what it offers."""
    return jsonify({
        "service": "realitystream",
        "endpoints": {
            "GET /health": "liveness probe",
            "GET /parameters": "preset YAML files in parameters/",
            "POST /run": 'body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}; ?upload=1 pushes the report',
        },
        "daily_cost_limit_usd": DAILY_COST_LIMIT_USD,
        "source": "https://github.com/ModelEarth/realitystream/blob/main/models/main.py",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "realitystream"}), 200


@app.route("/parameters", methods=["GET"])
def parameters():
    if not authorized():
        return jsonify({"status": "error", "message": "Missing or invalid X-API-Key"}), 401
    return jsonify({"parameters": preset_files()}), 200


@app.route("/run", methods=["POST", "OPTIONS"])
def run():
    if request.method == "OPTIONS":
        return "", 204
    if not authorized():
        return jsonify({"status": "error", "message": "Missing or invalid X-API-Key"}), 401

    if request.is_json:
        preset = (request.get_json(silent=True) or {}).get("parameters", "")
        if preset not in preset_files():
            return jsonify({"status": "error", "message": "Unknown parameters file", "available": preset_files()}), 400
        yaml_body = None
    else:
        yaml_body = request.get_data(as_text=True)
        if not yaml_body.strip():
            return jsonify({"status": "error", "message": "Empty request body; send a parameters.yaml"}), 400

    upload = request.args.get("upload", "").lower() in ("1", "true")
    today = date.today().isoformat()
    if _spent.get(today, 0) >= DAILY_COST_LIMIT_USD:
        return jsonify({"status": "throttled",
                        "message": f"Daily spending cap of ${DAILY_COST_LIMIT_USD:.2f} reached. Resets at midnight UTC."}), 429
    caller = caller_id()
    if throttled(caller):
        return jsonify({"status": "throttled", "caller": caller,
                        "message": f"Daily limit of {DAILY_RUNS_PER_USER} runs reached. Resets at midnight UTC."}), 429

    with tempfile.TemporaryDirectory(prefix="realitystream_") as work:
        if yaml_body is None:
            yaml_path = os.path.join(PARAMETERS_DIR, preset)
        else:
            yaml_path = os.path.join(work, "parameters.yaml")
            with open(yaml_path, "w", encoding="utf-8") as fh:
                fh.write(yaml_body)
        started = time.monotonic()
        try:
            from run_models import MissingKey, run_pipeline

            try:
                summary = run_pipeline(yaml_path, report_dir=os.path.join(work, "report"), upload=upload)
            except MissingKey as exc:
                return jsonify({"status": "missing_key", "key": exc.name, "how_to_get_it": exc.how_to_get_it}), 400
            summary.pop("report_dir", None)  # Temporary folder, removed after this request
            return jsonify({"status": "success", **summary}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            _spent[today] = _spent.get(today, 0) + (time.monotonic() - started) * COST_PER_SECOND_USD


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
