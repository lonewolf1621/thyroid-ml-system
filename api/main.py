"""FastAPI service for the hypothyroidism model.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from thyroid import __version__
from thyroid.drift import detect_drift
from thyroid.model import ThyroidModel
from thyroid.schema import FEATURES, PatientInput, patients_to_frame

STATIC_DIR = Path(__file__).parent / "static"
MAX_BATCH = 1000


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, default=str)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("thyroid.api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


class PredictionResponse(BaseModel):
    probability_hypothyroid: float = Field(description="Model probability that the patient is hypothyroid")
    prediction: str = Field(description="'hypothyroid' if probability >= threshold, else 'negative'")
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    patients: list[PatientInput] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


def create_app(model_dir: str | None = None) -> FastAPI:
    log = get_logger()
    state: dict = {"model": None, "error": None, "window": deque(maxlen=int(os.getenv("DRIFT_WINDOW", "500")))}
    prediction_log = os.getenv("PREDICTION_LOG_PATH")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path = model_dir or os.getenv("MODEL_DIR", "models")
        try:
            state["model"] = ThyroidModel.load(path)
            log.info("model loaded", extra={"extra_fields": {"model_dir": path, "model_version": state["model"].version}})
        except Exception as exc:  # keep serving /health so the failure is visible, not a crash loop
            state["error"] = f"{type(exc).__name__}: {exc}"
            log.error("model failed to load", extra={"extra_fields": {"model_dir": path, "error": state["error"]}})
        yield

    app = FastAPI(
        title="Thyroid ML System",
        version=__version__,
        description="Hypothyroidism risk prediction from a routine thyroid panel. "
                    "Portfolio project, not for clinical use.",
        lifespan=lifespan,
    )

    def require_model() -> ThyroidModel:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {state['error']}")
        return state["model"]

    def record(patients: list[PatientInput], results: list[dict], request_id: str) -> None:
        rows = [p.model_dump() for p in patients]
        state["window"].extend(rows)
        if prediction_log:
            Path(prediction_log).parent.mkdir(parents=True, exist_ok=True)
            with open(prediction_log, "a") as fh:
                for row, res in zip(rows, results, strict=False):
                    fh.write(json.dumps({"ts": time.time(), "request_id": request_id, "input": row, "output": res}) + "\n")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = str(latency_ms)
        if request.url.path not in ("/health", "/favicon.ico"):
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": response.status_code, "latency_ms": latency_ms}})
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        model = require_model()
        return {"status": "ready", "model_version": model.version}

    @app.get("/v1/model", tags=["model"])
    def model_info():
        model = require_model()
        m = model.metadata
        return {
            "model_version": m["model_version"],
            "model_type": m["model_type"],
            "created_utc": m["created_utc"],
            "threshold": m["threshold"],
            "target": m["target"],
            "features": FEATURES,
            "test_metrics": {k: v for k, v in m["test"].items() if k != "confusion_matrix"},
            "confusion_matrix": m["test"]["confusion_matrix"],
            "cv_results": m["cv"]["results"],
            "data": m["data"],
        }

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["model"])
    def predict(patient: PatientInput, request: Request):
        model = require_model()
        result = model.predict([patient])
        record([patient], result, request.state.request_id)
        return result[0]

    @app.post("/v1/predict/batch", response_model=BatchResponse, tags=["model"])
    def predict_batch(body: BatchRequest, request: Request):
        model = require_model()
        results = model.predict(body.patients)
        record(body.patients, results, request.state.request_id)
        return {"predictions": results}

    @app.get("/v1/monitoring/drift", tags=["monitoring"])
    def drift():
        """Compare the most recent requests (rolling window) with the training distribution."""
        model = require_model()
        window = list(state["window"])
        current = patients_to_frame([PatientInput(**row) for row in window]) if window else pd.DataFrame(columns=FEATURES)
        return detect_drift(model.reference, current)

    @app.post("/v1/predict-image", tags=["experimental"])
    async def predict_image(file: UploadFile = File(...)):
        """Experimental ultrasound classifier. Only available in builds that install PyTorch."""
        from thyroid import image

        if not image.available():
            raise HTTPException(status_code=503, detail="Image model is not enabled in this build (see README).")
        return image.predict(await file.read())

    return app


app = create_app()
