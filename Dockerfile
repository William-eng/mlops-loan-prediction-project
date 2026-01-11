# ---------------------------
# Stage 1: Builder
# ---------------------------
FROM python:3.10-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache-friendly)
COPY requirements_mlops.txt /app/

# Install Python packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_mlops.txt && \
    pip install --no-cache-dir azure-storage-blob fastapi uvicorn joblib pandas scikit-learn

# Copy source code
COPY main.py /app/main.py
COPY run_predict.py /app/run_predict.py
COPY prediction_model /app/prediction_model

# Copy the model file from local models directory
COPY models/loan_model.pkl /app/prediction_model/trained_models/model.pkl

COPY entrypoint.sh /app/entrypoint.sh

# ---------------------------
# Stage 2: Runtime
# ---------------------------
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:${PYTHONPATH:-}"

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST
RUN useradd -m -u 1000 appuser

# Create directories with proper ownership
RUN mkdir -p /app/prediction_model/trained_models \
             /app/prediction_model/datasets \
             /app/trained_models \
    && chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code with proper ownership
COPY --from=builder --chown=appuser:appuser /app/main.py /app/main.py
COPY --from=builder --chown=appuser:appuser /app/run_predict.py /app/run_predict.py
COPY --from=builder --chown=appuser:appuser /app/prediction_model /app/prediction_model
COPY --from=builder --chown=appuser:appuser /app/entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8005

# Environment variable for Azure Blob Storage (optional now, fallback only)
ENV AZURE_STORAGE_CONNECTION_STRING=""

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]