# syntax=docker/dockerfile:1
# Self-contained image: the model is trained from the public dataset at build time,
# so every build is reproducible from source (dataset hash and metrics end up in /v1/model).

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY thyroid ./thyroid

# ---- train ----------------------------------------------------------------
FROM base AS trainer
# Put data/hypothyroid.csv in the build context to build offline; otherwise it is downloaded from OpenML.
COPY data ./data
RUN python -m thyroid.data download --out data/hypothyroid.csv \
 && python -m thyroid.train --data data/hypothyroid.csv --out models

# ---- runtime --------------------------------------------------------------
FROM base AS runtime
COPY api ./api
COPY --from=trainer /app/models ./models
RUN useradd --create-home --uid 10001 app && mkdir -p /app/logs && chown app /app/logs
USER app
ENV PORT=8000 MODEL_DIR=/app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
