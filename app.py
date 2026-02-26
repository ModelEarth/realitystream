"""
RealityStream Cloud Run – Flask wrapper for the ML pipeline.

Exposes:
    POST /run   – Run the pipeline with a YAML config body.
    GET  /health – Liveness / readiness probe.

Deployment:
    gcloud run deploy realitystream --source .
or locally:
    python app.py
"""

import json
import os
import tempfile

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint for Cloud Run / load balancers."""
    return jsonify({"status": "ok"}), 200


@app.route("/run", methods=["POST"])
def run():
    """
    Accept a YAML configuration (as the request body) and run the
    RealityStream ML pipeline.  Returns JSON results.

    Example:
        curl -X POST http://localhost:8080/run \
             -H "Content-Type: text/yaml" \
             --data-binary @parameters/parameters-blinks.yaml
    """
    yaml_body = request.get_data(as_text=True)
    if not yaml_body.strip():
        return jsonify({"error": "Empty request body – send a YAML config"}), 400

    # Write to a temp file so run_pipeline can load it normally
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(yaml_body)
        tmp_path = tmp.name

    try:
        # Import here to keep startup fast
        from run_models import run_pipeline

        results = run_pipeline(tmp_path)

        # Strip the verbose text report for JSON response
        clean = []
        for r in results:
            clean.append({
                "model": r["model"],
                "accuracy": r["accuracy"],
                "roc_auc": r["roc_auc"],
                "duration_seconds": r["duration_seconds"],
                "classification_report": r["classification_report"],
            })
        return jsonify({"status": "success", "results": clean}), 200

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
