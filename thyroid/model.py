"""Load a trained model package and serve predictions (no web framework dependency)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from thyroid.schema import FEATURES, PatientInput, coerce_frame, patients_to_frame


@dataclass
class ThyroidModel:
    pipeline: object
    metadata: dict
    reference: dict

    @classmethod
    def load(cls, model_dir: str | Path) -> ThyroidModel:
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text())
        if metadata.get("features") != FEATURES:
            raise ValueError("Model was trained with a different feature schema than this code expects.")
        reference_path = model_dir / "drift_reference.json"
        reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
        pipeline = joblib.load(model_dir / "model.joblib")
        # Training uses all cores; for single-row requests, spinning up a worker pool costs more
        # than it saves (random forest: ~76 ms -> ~20 ms per prediction on a 2-vCPU box).
        estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
        if estimator is not None and hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        return cls(pipeline, metadata, reference)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        frame = coerce_frame(frame)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return [
            {
                "probability_hypothyroid": round(float(p), 4),
                "prediction": "hypothyroid" if p >= self.threshold else "negative",
                "threshold": round(self.threshold, 4),
                "model_version": self.version,
            }
            for p in proba
        ]

    def predict(self, patients: list[PatientInput]) -> list[dict]:
        return self.predict_frame(patients_to_frame(patients))
