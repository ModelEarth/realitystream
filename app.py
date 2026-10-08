"""
RealityStream Cloud Run: Flask wrapper around run_models.run_pipeline.

    GET  /health            liveness probe
    POST /run[?upload=1]    body = parameters.yaml text; returns the run summary as JSON

A missing key (for example GITHUB_REPORTS_TOKEN when upload=1) returns HTTP 400 with
`how_to_get_it`, which the front end shows to the user instead of a stack trace.
Each caller gets DAILY_RUNS_PER_USER runs a day (default 20); past that, HTTP 429.

Deploy:  gcloud run deploy realitystream --source .      (see deploy-cloud-run.sh)
Local:   python app.py
"""

import base64
import json
import os
import tempfile
from datetime import date

from flask import Flask, jsonify, request

app = Flask(__name__)

DAILY_RUNS_PER_USER = int(os.environ.get("DAILY_RUNS_PER_USER", "20"))
_runs = {}  # (day, caller) -> count. ponytail: per-instance memory, exact with --max-instances 1; use Firestore if that changes.


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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/run", methods=["POST"])
def run():
    yaml_body = request.get_data(as_text=True)
    if not yaml_body.strip():
        return jsonify({"status": "error", "message": "Empty request body; send a parameters.yaml"}), 400
    upload = request.args.get("upload", "").lower() in ("1", "true")
    caller = caller_id()
    if throttled(caller):
        return jsonify({"status": "throttled", "caller": caller,
                        "message": f"Daily limit of {DAILY_RUNS_PER_USER} runs reached. Resets at midnight UTC."}), 429

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write(yaml_body)
        yaml_path = tmp.name
    report_dir = tempfile.mkdtemp(prefix="report_")
    try:
        from run_models import MissingKey, run_pipeline

        try:
            summary = run_pipeline(yaml_path, report_dir=report_dir, upload=upload)
        except MissingKey as exc:
            return jsonify({"status": "missing_key", "key": exc.name, "how_to_get_it": exc.how_to_get_it}), 400
        return jsonify({"status": "success", **summary}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        os.unlink(yaml_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
