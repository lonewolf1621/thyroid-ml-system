import json
from pathlib import Path

import numpy as np

from tests.conftest import HEALTHY_PATIENT, HYPOTHYROID_PATIENT
from thyroid.model import ThyroidModel
from thyroid.schema import FEATURES, PatientInput
from thyroid.train import threshold_for_sensitivity


def test_training_writes_complete_package(model_dir):
    files = {p.name for p in Path(model_dir).iterdir()}
    assert {"model.joblib", "metadata.json", "drift_reference.json", "MODEL_CARD.md"} <= files
    meta = json.loads((Path(model_dir) / "metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert 0 < meta["threshold"] < 1
    assert set(meta["cv"]["results"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert meta["test"]["roc_auc"] > 0.9


def test_model_ranks_hypothyroid_above_healthy(model_dir):
    model = ThyroidModel.load(model_dir)
    hypo, healthy = model.predict([PatientInput(**HYPOTHYROID_PATIENT), PatientInput(**HEALTHY_PATIENT)])
    assert hypo["probability_hypothyroid"] > healthy["probability_hypothyroid"]
    assert hypo["prediction"] == "hypothyroid"
    assert healthy["prediction"] == "negative"


def test_prediction_label_follows_threshold(model_dir):
    model = ThyroidModel.load(model_dir)
    for result in model.predict([PatientInput(**HYPOTHYROID_PATIENT), PatientInput(**HEALTHY_PATIENT), PatientInput()]):
        expected = "hypothyroid" if result["probability_hypothyroid"] >= model.threshold else "negative"
        assert result["prediction"] == expected


def test_threshold_meets_target_sensitivity():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.15, len(y)), 0, 1)
    t = threshold_for_sensitivity(y, proba, 0.95)
    sensitivity = ((proba >= t) & (y == 1)).sum() / y.sum()
    assert sensitivity >= 0.95
