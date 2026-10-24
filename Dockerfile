# Multi-stage production Dockerfile for CacheSense
# Stage 1: Build & dependency resolution
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep image lightweight (<800MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download SentenceTransformer model into cache so cold start is instantaneous
ENV HF_HOME=/build/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build/hf_cache /home/cachesense/.cache/huggingface

# Create unprivileged runtime user
RUN groupadd -g 10001 cachesense && \
    useradd -u 10001 -g cachesense -m -s /bin/bash cachesense && \
    mkdir -p /app/data && \
    chown -R cachesense:cachesense /app /home/cachesense

# Copy application source
COPY --chown=cachesense:cachesense app /app/app
COPY --chown=cachesense:cachesense dashboard /app/dashboard
COPY --chown=cachesense:cachesense benchmarks /app/benchmarks
COPY --chown=cachesense:cachesense pyproject.toml /app/pyproject.toml

USER cachesense

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SQLITE_DB_PATH=/app/data/cache.db \
    FAISS_INDEX_PATH=/app/data/faiss.index \
    HOST=0.0.0.0 \
    PORT=8000

VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
