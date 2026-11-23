import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from tests.conftest import HEALTHY_PATIENT, HYPOTHYROID_PATIENT  # noqa: E402


@pytest.fixture(scope="module")
def client(model_dir):
    with TestClient(create_app(model_dir)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["model_version"]


def test_predict_hypothyroid(client):
    res = client.post("/v1/predict", json=HYPOTHYROID_PATIENT)
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "hypothyroid"
    assert 0 <= body["probability_hypothyroid"] <= 1
    assert res.headers["x-request-id"]


def test_predict_validation_error(client):
    assert client.post("/v1/predict", json={"age": -3}).status_code == 422
    assert client.post("/v1/predict", json={"on thyroxine": "t"}).status_code == 422


def test_batch(client):
    res = client.post("/v1/predict/batch", json={"patients": [HYPOTHYROID_PATIENT, HEALTHY_PATIENT]})
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert [p["prediction"] for p in preds] == ["hypothyroid", "negative"]


def test_model_info_and_drift(client):
    info = client.get("/v1/model").json()
    assert info["model_type"] and info["threshold"] > 0
    drift = client.get("/v1/monitoring/drift").json()
    assert drift["status"] in {"insufficient_data", "ok", "drift"}


def test_ui_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Thyroid ML System" in res.text


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/v1/predict", json=HEALTHY_PATIENT).status_code == 503
