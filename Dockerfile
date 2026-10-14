# ─────────────────────────────────────────────────────────────────────────
# Dockerfile — Traffic Accident Severity Prediction API
# ─────────────────────────────────────────────────────────────────────────
# Build:  docker build -t accident-api .
# Run:    docker run -p 8000:8000 -v %cd%/models:/app/models accident-api
#
# The image does NOT bundle the virtual environment or data files.
# Mount the models/ directory at runtime to provide the trained model.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install OS deps required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the API and model-loading code
COPY api/         ./api/
COPY scripts/train_model.py ./scripts/

# Create directories that will be volume-mounted
RUN mkdir -p models logs data/cleaned

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check — calls /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Read configuration from environment variables
ENV MODEL_PATH=models/severity_model.pkl
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
