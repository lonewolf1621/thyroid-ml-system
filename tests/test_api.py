"""API tests: run with `pytest -q` (needs requirements.txt + requirements-dev.txt)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)

NORMAL = {"age": 45, "sex": "F", "TSH": 1.3, "T3": 2.0, "TT4": 110, "T4U": 1.0, "FTI": 110}
HYPO = {"age": 55, "sex": "F", "TSH": 60, "T3": 0.5, "TT4": 40, "T4U": 1.0, "FTI": 40}


def test_health_lists_all_models():
    r = client.get("/health")
    assert r.status_code == 200
    assert {"RandomForest", "XGBoost", "LogisticRegression"} <= set(r.json()["tabular_models"])


def test_demo_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "Thyroid ML System" in r.text


@pytest.mark.parametrize("model", ["RandomForest", "XGBoost", "LogisticRegression"])
def test_predictions_follow_lab_values(model):
    normal = client.post(f"/predict?model_name={model}", json=NORMAL).json()
    hypo = client.post(f"/predict?model_name={model}", json=HYPO).json()
    assert normal["prediction"] == "negative"
    assert hypo["prediction"] == "hypothyroid"
    assert hypo["probability_hypothyroid"] > normal["probability_hypothyroid"]


def test_invalid_input_rejected():
    assert client.post("/predict", json={**NORMAL, "sex": "X"}).status_code == 422
    assert client.post("/predict", json={**NORMAL, "TSH": -1}).status_code == 422


def test_unknown_model_404():
    assert client.post("/predict?model_name=Nope", json=NORMAL).status_code == 404


def test_image_prediction_shape():
    buf = io.BytesIO()
    Image.new("L", (300, 200), color=90).save(buf, "PNG")
    r = client.post("/predict-image", files={"file": ("scan.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"benign", "malignant"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_non_image_rejected():
    r = client.post("/predict-image", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_model_card_endpoint():
    assert client.get("/model-card").status_code == 200
