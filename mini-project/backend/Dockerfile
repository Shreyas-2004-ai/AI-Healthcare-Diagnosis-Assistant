# ─────────────────────────────────────────────
# Stage 1: Builder — install dependencies
# ─────────────────────────────────────────────
FROM python:3.12.0-slim AS builder

# System deps needed to compile some Python packages (e.g. numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --upgrade pip==24.0 \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────
FROM python:3.12.0-slim

# Runtime system lib for numpy/scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
# These are defaults — override at runtime via docker run -e or docker-compose
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# ─────────────────────────────────────────────
# Expose & run
# ─────────────────────────────────────────────
EXPOSE 5000

# gunicorn 21.2.0 — production WSGI server (matches requirements.txt)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
