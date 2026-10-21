"""
RealityStream serverless API for Google Cloud Run: Flask wrapper around run_models.run_pipeline.

    GET  /                     home page (api/home.html): settings, Run button, today's run time and cost;
                               JSON for non-browsers
    GET  /health               liveness probe
    GET  /key                  whether header X-API-Key is valid
    GET  /parameters           preset YAML files in parameters/
    GET  /github-token         whether GITHUB_REPORTS_TOKEN can push to modelearth/reports (asked of GitHub, cached 5 minutes);
                               header X-GitHub-Token checks a token saved in the browser (keys/#need=GITHUB_REPORTS_TOKEN) instead
    GET  /model-params         settings the YAML's split, search and hyperparameters sections may set
                               (api/model-params.yaml), for the Common Settings panel and each model's fields
    POST /run[?upload=1][&smote=0|1]
                               body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}
                               returns the run summary as JSON; smote=0 trains only without SMOTE,
                               smote=1 only with SMOTE, omitted trains both

Each run's report folder is named {year}/run-{date}T{time}. Run locally (not on Cloud Run), it's saved in
the webroot's reports folder (REPORTS_DIR overrides), and `output_path` (for example
reports/2026/run-2026-10-21T14-05-00/) is its path from the webroot. With the Team Passphrase, the folder is
also pushed to GitHub modelearth/reports when GITHUB_REPORTS_TOKEN is set; `github.message` says whether it
was. upload=1 requires the push (HTTP 400 without the token). Header X-GitHub-Token, a token saved in the
browser, is pushed with in place of the setting, with or without the Team Passphrase.

A YAML value that model-params.yaml doesn't allow returns HTTP 400 with `field` naming it, before the
run is counted. A missing key (for example GITHUB_REPORTS_TOKEN when upload=1) returns HTTP 400 with
`how_to_get_it`, which the front end shows to the user instead of a stack trace.
With the Team Passphrase (header X-API-Key matching REALITYSTREAM_API_KEY, shared with the team; not a Google key), a caller gets DAILY_RUNS_PER_USER
runs a day (default 20) with any number of models, may change model settings, and may use upload=1.
Without a key, a caller gets FREE_RUNS_PER_DAY run (default 1) of 1 model with default settings; past that,
HTTP 429 with how to run more.
Callers are identified by IP address. Runs also stop for the day once their estimated cost reaches
DAILY_COST_LIMIT_USD (default $0.20).
Daily totals (run seconds, estimated cost, runs) are kept in Firestore on Cloud Run, so redeploys
and restarts don't reset them; local runs keep them in memory. Days run midnight to midnight
in USAGE_TIMEZONE (default America/New_York, Eastern Time).

Settings come from run_models.get_env: environment, then the OS credential store, then the
env file named in automation/paths.yaml. On Cloud Run, set them with --set-env-vars or --set-secrets.

Deploy (no Dockerfile, Google Cloud buildpacks read requirements.txt and Procfile):
    cd realitystream && ./deploy/deploy-cloud-run.sh
Local (from the realitystream folder):
    python api/main.py
"""

import hashlib
import hmac
import os
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request

REALITYSTREAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETERS_DIR = os.path.join(REALITYSTREAM_DIR, "parameters")
WEBROOT_DIR = os.path.dirname(REALITYSTREAM_DIR)
# Local runs save their report folders here (the webroot's reports repo), so the page can link to them
ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))  # Cloud Run sets K_SERVICE
REPORTS_DIR = os.environ.get("REPORTS_DIR") or os.path.join(WEBROOT_DIR, "reports")

# run_models.py and keystore.py reside beside this file, in realitystream/api/
API_DIR = os.path.dirname(os.path.abspath(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

app = Flask(__name__)

DAILY_RUNS_PER_USER = int(os.environ.get("DAILY_RUNS_PER_USER", "20"))
# Without the Team Passphrase: this many runs a day per caller, of one model each
FREE_RUNS_PER_DAY = int(os.environ.get("FREE_RUNS_PER_DAY", "1"))
MORE_RUNS_URL = "https://model.earth/realitystream/deploy/#googleaccountautomation"
MORE_RUNS = ("To run more, add a REALITYSTREAM_API_KEY for this Google Cloud project, "
             "or run on your own cloud account.")

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

# Pages that call this API from another site: realitystream/models/ on model.earth and the local webroot
ALLOWED_ORIGINS = {"https://model.earth", "http://localhost:8887", "http://127.0.0.1:8887"}


@app.after_request
def add_cors_headers(response):
    """Allow model.earth and the local webroot (port 8887) to call this API."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, X-GitHub-Token, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def api_key_enabled():
    from run_models import get_env

    return bool(get_env("REALITYSTREAM_API_KEY", ""))


def key_status():
    """'valid' when X-API-Key matches REALITYSTREAM_API_KEY (or no key is configured), 'missing' or 'invalid' otherwise."""
    from run_models import get_env

    expected = get_env("REALITYSTREAM_API_KEY", "")
    if not expected:
        return "valid"
    sent = request.headers.get("X-API-Key", "")
    if not sent:
        return "missing"
    return "valid" if hmac.compare_digest(sent, expected) else "invalid"


def github_token_set():
    """Whether GITHUB_REPORTS_TOKEN is set (never its value), for the page's message about pushing reports."""
    from run_models import get_env

    return bool(get_env("GITHUB_REPORTS_TOKEN", ""))


def saves_locally():
    """Local runs keep their report folders in REPORTS_DIR; Cloud Run's are temporary."""
    return not ON_CLOUD_RUN and os.path.isdir(REPORTS_DIR)


def caller_id():
    """Who is calling: the client IP address. Google's front end appends it as the last X-Forwarded-For
    entry; earlier entries can be set by the client, so they aren't trusted."""
    forwarded = [ip.strip() for ip in request.headers.get("X-Forwarded-For", "").split(",") if ip.strip()]
    return forwarded[-1] if forwarded else (request.remote_addr or "anon")


def preset_files():
    return sorted(f for f in os.listdir(PARAMETERS_DIR) if f.endswith((".yaml", ".yml")))


ENDPOINTS = {
    "GET /health": "Liveness probe",
    "GET /key": "Whether header X-API-Key is valid",
    "GET /parameters": "Preset YAML files in parameters/",
    "GET /github-token": "Whether GITHUB_REPORTS_TOKEN (or header X-GitHub-Token) can push to modelearth/reports",
    "GET /model-params": "Settings the YAML's split, search and hyperparameters sections may set",
    "POST /run": 'Body = parameters.yaml text, or JSON {"parameters": "parameters-blinks.yaml"}. '
                 'Add ?smote=0 to train only without SMOTE, ?smote=1 only with SMOTE (omit to train both), '
                 'Reports are pushed to GitHub when GITHUB_REPORTS_TOKEN is set (needs the key); ?upload=1 requires it. '
                 'Without header X-API-Key: 1 model, once per day.',
}
SOURCE_URL = "https://github.com/ModelEarth/realitystream/blob/main/api/main.py"

HOME_PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home.html")
# Webroot the home page loads shared files from (param-input.js, model-select.js, localsite.js).
# Set ASSET_BASE=http://localhost:8887/ to test local changes to those files.
ASSET_BASE = os.environ.get("ASSET_BASE", "https://model.earth/")


@app.route("/", methods=["GET"])
def index():
    """Home page (api/home.html) for browsers; JSON with today's totals and the endpoints otherwise."""
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
                    "api_key_enabled": api_key_enabled(), "free_runs_per_day": FREE_RUNS_PER_DAY,
                    "github_token": github_token_set(), "saves_locally": saves_locally(),
                    "more_runs": MORE_RUNS, "more_runs_url": MORE_RUNS_URL,
                    "endpoints": ENDPOINTS, "source": SOURCE_URL}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "realitystream"}), 200


@app.route("/key", methods=["GET"])
def key():
    """Whether header X-API-Key is valid ('valid', 'missing' or 'invalid'), so pages can show the right limits."""
    return jsonify({"key": key_status(), "api_key_enabled": api_key_enabled()}), 200


BROWSER_TOKEN_LABEL = "GITHUB_REPORTS_TOKEN (from this browser)"


def browser_github_token():
    """The GitHub token the page sends from the browser's keys (header X-GitHub-Token), or ''. Never logged."""
    return request.headers.get("X-GitHub-Token", "").strip()


@app.route("/parameters", methods=["GET"])
def parameters():
    return jsonify({"parameters": preset_files()}), 200


# GitHub's answer about each token (by hash, so a changed token is checked again), for 5 minutes
GITHUB_CHECK_SECONDS = 300
GITHUB_CHECKS_KEPT = 100
_github_checks = {}


@app.route("/github-token", methods=["GET"])
def github_token():
    """Whether GITHUB_REPORTS_TOKEN, or the browser's token (header X-GitHub-Token), can push to the reports
    repo, for the page's line after Run Models. source: "browser" or "server"."""
    from run_models import check_reports_token, get_env

    sent = browser_github_token()
    token = sent or get_env("GITHUB_REPORTS_TOKEN", "") or ""
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.monotonic()
    cached = _github_checks.get(key)
    if not cached or now - cached[0] > GITHUB_CHECK_SECONDS:
        cached = (now, check_reports_token(token, BROWSER_TOKEN_LABEL if sent else "GITHUB_REPORTS_TOKEN"))
        if cached[1]["token"] != "unknown":  # Check again next time when GitHub couldn't answer
            for old in [k for k, v in _github_checks.items() if now - v[0] > GITHUB_CHECK_SECONDS]:
                del _github_checks[old]
            if len(_github_checks) >= GITHUB_CHECKS_KEPT:
                _github_checks.clear()
            _github_checks[key] = cached
    return jsonify({**cached[1], "source": "browser" if sent else "server"}), 200


@app.route("/model-params", methods=["GET"])
def model_params():
    """api/model-params.yaml as JSON: each setting's type, default, range or options, and help text."""
    from run_models import model_params_schema

    return jsonify(model_params_schema()), 200


@app.route("/run", methods=["POST", "OPTIONS"])
def run():
    if request.method == "OPTIONS":
        return "", 204
    status = key_status()
    if status == "invalid":
        return jsonify({"status": "error", "message": "Team Passphrase not accepted (header X-API-Key)"}), 401
    keyed = status == "valid"

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

    # Check the YAML before the run is counted, so a mistake doesn't use up a run
    from run_models import SETTINGS_SECTIONS, ParameterError, load_parameters, validate_parameters

    try:
        if yaml_body is None:
            params = load_parameters(os.path.join(PARAMETERS_DIR, preset))
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                tmp.write(yaml_body)
            try:
                params = load_parameters(tmp.name)
            finally:
                os.unlink(tmp.name)
        validate_parameters(params)
    except ParameterError as exc:
        return jsonify({"status": "error", "field": exc.field, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Invalid parameters: {exc}"}), 400

    if not keyed:
        # Without the Team Passphrase: 1 model with default settings, no report upload
        if len(params["models"] or ["rfc"]) > 1:
            return jsonify({"status": "key_needed", "more_runs_url": MORE_RUNS_URL,
                            "message": f"Without the Team Passphrase, choose 1 model. {MORE_RUNS}"}), 403
        changed = [section for section in SETTINGS_SECTIONS if params.get(section)]
        if changed:
            return jsonify({"status": "key_needed", "more_runs_url": MORE_RUNS_URL, "sections": changed,
                            "message": f"Changing model settings ({', '.join(changed)}) needs the Team Passphrase. "
                                       f"{MORE_RUNS}"}), 403
        if upload:
            return jsonify({"status": "key_needed", "more_runs_url": MORE_RUNS_URL,
                            "message": f"Uploading reports needs the Team Passphrase. {MORE_RUNS}"}), 403

    today = usage.today()
    caller = caller_id()
    # Keyless runs are counted separately, so a key holder's runs from the same address don't use them up
    count_as = caller if keyed else "free:" + caller
    run_limit = DAILY_RUNS_PER_USER if keyed else FREE_RUNS_PER_DAY
    try:
        spent = usage.totals(today)["cost_usd"]
        caller_runs = usage.caller_runs(count_as, today)
    except Exception:
        # Without the daily totals the cap can't be enforced, so don't start a run
        return jsonify({"status": "error", "message": "Usage totals are unavailable; try again shortly."}), 503
    if spent >= DAILY_COST_LIMIT_USD:
        return jsonify({"status": "throttled",
                        "message": f"Daily limit of {DAILY_LIMIT_MINUTES:.0f} minutes (${DAILY_COST_LIMIT_USD:.2f}) reached. Resets at midnight Eastern Time."}), 429
    if caller_runs >= run_limit:
        if keyed:
            return jsonify({"status": "throttled", "caller": caller,
                            "message": f"Daily limit of {DAILY_RUNS_PER_USER} runs reached. Resets at midnight Eastern Time."}), 429
        return jsonify({"status": "throttled", "caller": caller, "more_runs_url": MORE_RUNS_URL,
                        "message": f"Without the Team Passphrase, you can run 1 model once per day. {MORE_RUNS}"}), 429
    usage.count_run(count_as, today)

    with tempfile.TemporaryDirectory(prefix="realitystream_") as work:
        if yaml_body is None:
            yaml_path = os.path.join(PARAMETERS_DIR, preset)
        else:
            yaml_path = os.path.join(work, "parameters.yaml")
            with open(yaml_path, "w", encoding="utf-8") as fh:
                fh.write(yaml_body)
        started = time.monotonic()
        try:
            from run_models import MissingKey, report_name, run_pipeline

            name = report_name()
            local = saves_locally()
            report_dir = os.path.join(REPORTS_DIR, *name.split("/")) if local else os.path.join(work, "report")
            # Pushed to GitHub when a token is available (upload=1 requires it). Keyless runs are pushed only
            # with the browser's own token, not the service's.
            sent = browser_github_token()
            push = True if upload else ("auto" if keyed or sent else False)
            try:
                summary = run_pipeline(yaml_path, report_dir=report_dir, upload=push, smote=smote, name=name,
                                       token=sent or None,
                                       token_label=BROWSER_TOKEN_LABEL if sent else "GITHUB_REPORTS_TOKEN")
            except MissingKey as exc:
                return jsonify({"status": "missing_key", "key": exc.name, "how_to_get_it": exc.how_to_get_it}), 400
            summary.pop("report_dir", None)  # Not shown: on Cloud Run, a temporary folder removed after this request
            summary["report_name"] = name
            summary["output_path"] = None
            if local:
                rel = os.path.relpath(report_dir, WEBROOT_DIR)
                summary["output_path"] = None if rel.startswith("..") else rel.replace(os.sep, "/") + "/"
            if not push:
                summary["github"] = {"token": None, "pushed_to": None,
                                     "message": "Reports are pushed to GitHub only with the Team Passphrase "
                                                "or a GITHUB_REPORTS_TOKEN saved in this browser."}
            return jsonify({"status": "success", "keyed": keyed, **summary}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            usage.add_time(time.monotonic() - started, today)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
