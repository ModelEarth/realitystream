# -----------------------------------------------------------
# RealityStream ML – Cloud Run container
# -----------------------------------------------------------
# Build:  docker build -t realitystream .
# Run:    docker run -p 8080:8080 realitystream
# Deploy: gcloud run deploy realitystream --source .
# -----------------------------------------------------------

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run_models.py app.py ./
COPY parameters/ parameters/

# Cloud Run uses PORT env variable (default 8080)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 300 app:app
