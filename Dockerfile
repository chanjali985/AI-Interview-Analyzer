# ---------- stage 1: build the React frontend ----------
FROM node:20-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# ---------- stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    FRONTEND_DIST=/app/frontend/dist

# ffmpeg decodes the browser's webm/opus recordings; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install --upgrade pip && pip install -r requirements-ml.txt

COPY backend/ /app/
COPY --from=frontend /build/dist /app/frontend/dist

# Warm the Whisper model into the image so the first interview is not slow.
# Needs network access to huggingface.co at build time; set SKIP_MODEL_WARMUP=1
# to build without it (the model is then downloaded on first use instead).
ARG SKIP_MODEL_WARMUP=0
ARG WHISPER_MODEL=base
RUN if [ "$SKIP_MODEL_WARMUP" = "0" ]; then \
      python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')" \
      && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
      || echo 'Model warmup skipped (no network at build time)'; \
    fi

RUN mkdir -p /app/data/uploads && chown -R app:app /app /home/app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health/live || exit 1

# One web worker by default: analysis runs in this process's thread pool, so a
# second worker would race on requeueing unfinished interviews. Scale with
# ANALYSIS_WORKERS, or move the queue to Celery/RQ before scaling horizontally.
ENV WEB_CONCURRENCY=1
CMD ["sh", "-c", "gunicorn app.main:app --workers ${WEB_CONCURRENCY} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile -"]
