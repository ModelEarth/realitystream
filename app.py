"""
RealityStream Cloud Run: Flask wrapper around run_models.run_pipeline.

    GET  /health            liveness probe
    POST /run[?upload=1]    body = parameters.yaml text; returns the run summary as JSON

A missing key (for example GITHUB_REPORTS_TOKEN when upload=1) returns HTTP 400 with
`how_to_get_it`, which the front end shows to the user instead of a stack trace.

Deploy:  gcloud run deploy realitystream --source .      (see deploy-cloud-run.sh)
Local:   python app.py
"""

import os
import tempfile

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/run", methods=["POST"])
def run():
    yaml_body = request.get_data(as_text=True)
    if not yaml_body.strip():
        return jsonify({"status": "error", "message": "Empty request body; send a parameters.yaml"}), 400
    upload = request.args.get("upload", "").lower() in ("1", "true")

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
