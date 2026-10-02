# ============================================================
# Explainable Hybrid Fraud Detection System
# Multi-stage Docker build
# ============================================================

# ---- Base Stage ----
FROM python:3.11-slim AS base

LABEL maintainer="Fraud Detection System"
LABEL description="Explainable Hybrid Fraud Detection using XGBoost, Isolation Forest & SHAP"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies Stage ----
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Application Stage ----
FROM dependencies AS app

COPY . .

# Create necessary directories
RUN mkdir -p data models reports experiments mlruns reports/logs

# Expose ports
EXPOSE 8000 8501

# Default command: run training then start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
