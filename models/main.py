"""
RealityStream serverless API for Google Cloud Run: Flask wrapper around run_models.run_pipeline.

    GET  /                     home page (models/home.html): settings, Run button, today's run time and cost;
                               JSON for non-browsers
    GET  /health               liveness probe
    GET  /parameters           preset YAML files in parameters/
    POST /run[?upload=1][&smote=0|1]
                               body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}
                               returns the run summary as JSON; upload=1 pushes the report folder;
                               smote=0 trains only without SMOTE, smote=1 only with SMOTE, omitted trains both

A missing key (for example GITHUB_REPORTS_TOKEN when upload=1) returns HTTP 400 with
`how_to_get_it`, which the front end shows to the user instead of a stack trace.
Each caller gets DAILY_RUNS_PER_USER runs a day (default 20); past that, HTTP 429.
Runs also stop for the day once their estimated cost reaches DAILY_COST_LIMIT_USD (default $0.20).
Daily totals (run seconds, estimated cost, runs) are kept in Firestore on Cloud Run, so redeploys
and restarts don't reset them; local runs keep them in memory. Days run midnight to midnight
in USAGE_TIMEZONE (default America/New_York, Eastern Time).
When REALITYSTREAM_API_KEY is set, /parameters and /run require header X-API-Key.

Settings come from run_models.get_env: environment, then the OS credential store, then the
env file named in automation/paths.yaml. On Cloud Run, set them with --set-env-vars or --set-secrets.

Deploy (no Dockerfile, Google Cloud buildpacks read requirements.txt and Procfile):
    cd realitystream && ./deploy-cloud-run.sh
Local (from the realitystream folder):
    python models/main.py
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request

REALITYSTREAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETERS_DIR = os.path.join(REALITYSTREAM_DIR, "parameters")

# run_models.py resides in the realitystream folder
if REALITYSTREAM_DIR not in sys.path:
    sys.path.insert(0, REALITYSTREAM_DIR)

app = Flask(__name__)

DAILY_RUNS_PER_USER = int(os.environ.get("DAILY_RUNS_PER_USER", "20"))

# Daily spending cap. Run time is priced at Cloud Run request-based rates for 4 vCPU + 2 GiB
# (4 x $0.000024 per vCPU-second + 2 x $0.0000025 per GiB-second), ignoring the free tier.
DAILY_COST_LIMIT_USD = float(os.environ.get("DAILY_COST_LIMIT_USD", "0.20"))
COST_PER_SECOND_USD = float(os.environ.get("COST_PER_SECOND_USD", "0.000101"))
DAILY_LIMIT_MINUTES = DAILY_COST_LIMIT_USD / COST_PER_SECOND_USD / 60
# Daily totals reset at midnight in this time zone
USAGE_TIMEZONE = ZoneInfo(os.environ.get("USAGE_TIMEZONE", "America/New_York"))


class UsageStore:
    """Daily totals in Firestore (collection "usage", one document per day in USAGE_TIMEZONE), or in memory locally.

    Each day document holds seconds, cost_usd and runs. Per-caller run counts sit in its "callers"
    subcollection under a hash of the caller, so emails and IP addresses are not stored.
    """

    def __init__(self):
        self._db = None
        self._memory = {}
        if os.environ.get("K_SERVICE") or os.environ.get("USAGE_STORE") == "firestore":
            from google.cloud import firestore  # Cloud Run sets K_SERVICE
            self._firestore = firestore
            self._db = firestore.Client()

    @staticmethod
    def today():
        return datetime.now(USAGE_TIMEZONE).date().isoformat()

    @staticmethod
    def _caller_key(caller):
        return hashlib.sha256(caller.encode("utf-8")).hexdigest()[:16]

    def _day(self, day):
        return self._db.collection("usage").document(day)

    def totals(self, day=None):
        """{seconds, cost_usd, runs} for the day."""
        day = day or self.today()
        if self._db is None:
            data = self._memory.get(day, {})
        else:
            snap = self._day(day).get()
            data = (snap.to_dict() or {}) if snap.exists else {}
        return {"seconds": data.get("seconds", 0.0), "cost_usd": data.get("cost_usd", 0.0), "runs": data.get("runs", 0)}

    def caller_runs(self, caller, day=None):
        day = day or self.today()
        key = self._caller_key(caller)
        if self._db is None:
            return self._memory.get(day, {}).get("callers", {}).get(key, 0)
        snap = self._day(day).collection("callers").document(key).get()
        return (snap.to_dict() or {}).get("runs", 0) if snap.exists else 0

    def count_run(self, caller, day=None):
        """Count a started run for the day and for this caller."""
        day = day or self.today()
        key = self._caller_key(caller)
        if self._db is None:
            totals = self._memory.setdefault(day, {})
            totals["runs"] = totals.get("runs", 0) + 1
            callers = totals.setdefault("callers", {})
            callers[key] = callers.get(key, 0) + 1
            return
        inc = self._firestore.Increment(1)
        self._day(day).set({"runs": inc}, merge=True)
        self._day(day).collection("callers").document(key).set({"runs": inc}, merge=True)

    def add_time(self, seconds, day=None):
        """Add finished run time, and its estimated cost, to the day's totals."""
        day = day or self.today()
        cost = seconds * COST_PER_SECOND_USD
        if self._db is None:
            totals = self._memory.setdefault(day, {})
            totals["seconds"] = totals.get("seconds", 0.0) + seconds
            totals["cost_usd"] = totals.get("cost_usd", 0.0) + cost
            return
        self._day(day).set({"seconds": self._firestore.Increment(seconds),
                            "cost_usd": self._firestore.Increment(cost)}, merge=True)


usage = UsageStore()

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


def preset_files():
    return sorted(f for f in os.listdir(PARAMETERS_DIR) if f.endswith((".yaml", ".yml")))


ENDPOINTS = {
    "GET /health": "Liveness probe",
    "GET /parameters": "Preset YAML files in parameters/",
    "POST /run": 'Body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}. '
                 'Add ?smote=0 to train only without SMOTE, ?smote=1 only with SMOTE (omit to train both), '
                 '?upload=1 to push the report.',
}
SOURCE_URL = "https://github.com/ModelEarth/realitystream/blob/main/models/main.py"

HOME_PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home.html")
# Webroot the home page loads shared files from (param-input.js, model-select.js, localsite.js).
# Set ASSET_BASE=http://localhost:8887/ to test local changes to those files.
ASSET_BASE = os.environ.get("ASSET_BASE", "https://model.earth/")


def api_key_required():
    from run_models import get_env

    return bool(get_env("REALITYSTREAM_API_KEY", ""))


@app.route("/", methods=["GET"])
def index():
    """Home page (models/home.html) for browsers; JSON with today's totals and the endpoints otherwise."""
    if "text/html" in request.headers.get("Accept", ""):
        with open(HOME_PAGE_FILE, encoding="utf-8") as fh:
            return fh.read().replace("{{ASSET_BASE}}", ASSET_BASE), 200, {"Content-Type": "text/html; charset=utf-8"}
    day = usage.today()
    try:
        totals = usage.totals(day)
    except Exception:
        totals = None
    # Minutes left = remaining budget at the current per-second rate
    remaining = max(0.0, (DAILY_COST_LIMIT_USD - (totals or {}).get("cost_usd", 0.0)) / COST_PER_SECOND_USD / 60)
    return jsonify({"service": "realitystream", "day": day, "today": totals,
                    "minutes_left_today": round(remaining, 1) if totals else None,
                    "daily_cost_limit_usd": DAILY_COST_LIMIT_USD, "daily_limit_minutes": round(DAILY_LIMIT_MINUTES, 1),
                    "api_key_required": api_key_required(),
                    "endpoints": ENDPOINTS, "source": SOURCE_URL}), 200


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
    smote_arg = request.args.get("smote", "").lower()
    if smote_arg not in ("", "0", "1", "false", "true"):
        return jsonify({"status": "error", "message": "smote must be 0 or 1 (omit to train both)"}), 400
    smote = None if smote_arg == "" else smote_arg in ("1", "true")
    today = usage.today()
    caller = caller_id()
    try:
        spent = usage.totals(today)["cost_usd"]
        caller_runs = usage.caller_runs(caller, today)
    except Exception:
        # Without the daily totals the cap can't be enforced, so don't start a run
        return jsonify({"status": "error", "message": "Usage totals are unavailable; try again shortly."}), 503
    if spent >= DAILY_COST_LIMIT_USD:
        return jsonify({"status": "throttled",
                        "message": f"Daily limit of {DAILY_LIMIT_MINUTES:.0f} minutes (${DAILY_COST_LIMIT_USD:.2f}) reached. Resets at midnight Eastern Time."}), 429
    if caller_runs >= DAILY_RUNS_PER_USER:
        return jsonify({"status": "throttled", "caller": caller,
                        "message": f"Daily limit of {DAILY_RUNS_PER_USER} runs reached. Resets at midnight Eastern Time."}), 429
    usage.count_run(caller, today)

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
                summary = run_pipeline(yaml_path, report_dir=os.path.join(work, "report"), upload=upload, smote=smote)
            except MissingKey as exc:
                return jsonify({"status": "missing_key", "key": exc.name, "how_to_get_it": exc.how_to_get_it}), 400
            summary.pop("report_dir", None)  # Temporary folder, removed after this request
            return jsonify({"status": "success", **summary}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            usage.add_time(time.monotonic() - started, today)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
