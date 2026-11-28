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


def test_image_gradcam_heatmap():
    import base64
    buf = io.BytesIO()
    Image.new("L", (300, 200), color=90).save(buf, "PNG")
    r = client.post("/predict-image?explain=true", files={"file": ("scan.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    png = base64.b64decode(r.json()["gradcam_png_base64"])
    assert Image.open(io.BytesIO(png)).size == (300, 200)


def test_batch_template_and_scoring():
    tpl = client.get("/predict/batch/template")
    assert tpl.status_code == 200 and tpl.text.startswith("age,sex")
    r = client.post("/predict/batch", files={"file": ("p.csv", tpl.content, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scored"] == 2 and not body["errors"]
    assert [x["prediction"] for x in body["results"]] == ["negative", "hypothyroid"]


def test_batch_reports_bad_rows_and_csv_output():
    csv = ("age,sex,TSH,T3,TT4,T4U,FTI\n"
           "45,F,1.3,2.0,110,1.0,110\n"
           "200,X,1,1,1,1,1\n"
           "55,F,60,0.5,40,1.0,40\n")
    r = client.post("/predict/batch", files={"file": ("p.csv", csv.encode(), "text/csv")})
    body = r.json()
    assert body["rows"] == 3 and body["scored"] == 2
    assert body["errors"][0]["row"] == 2
    out = client.post("/predict/batch?output=csv", files={"file": ("p.csv", csv.encode(), "text/csv")})
    assert out.status_code == 200 and "error" in out.text


def test_batch_rejects_non_csv():
    r = client.post("/predict/batch", files={"file": ("p.csv", b"", "text/csv")})
    assert r.status_code == 400


# ----------------------------- owner admin
ADMIN_TOKEN = "test-admin-token-" + "y" * 24
AH = {"X-Admin-Token": ADMIN_TOKEN}


def _admin_env(monkeypatch):
    from api import admin
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    admin._failures.clear()
    admin.settings.update(admin.DEFAULT_SETTINGS)
    return admin


def test_admin_requires_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.get("/admin/api/stats", headers=AH).status_code == 403
    admin = _admin_env(monkeypatch)
    assert client.get("/admin/api/stats").status_code == 401
    assert client.get("/admin/api/stats", headers={"X-Admin-Token": "nope" * 8}).status_code == 401
    assert client.get("/admin").status_code == 200
    admin._failures.clear()


def test_admin_stats_count_predictions_and_logs_hide_inputs(monkeypatch):
    _admin_env(monkeypatch)
    client.post("/predict", json=NORMAL)
    stats = client.get("/admin/api/stats", headers=AH).json()
    assert any(k.startswith("tabular:RandomForest:") for k in stats["counters"])
    assert any(e["endpoint"] == "POST /predict" for e in stats["endpoints"])
    items = client.get("/admin/api/logs?kind=events", headers=AH).json()["items"]
    ev = next(i for i in items if i["type"] == "prediction_tabular")
    assert "input" not in ev and "TSH" not in str(ev)


def test_admin_can_switch_features_off(monkeypatch):
    admin = _admin_env(monkeypatch)
    r = client.put("/admin/api/settings", headers=AH, json={"maintenance_banner": "Model update tonight", "image_enabled": False})
    assert r.status_code == 200
    assert client.get("/status").json()["maintenance_banner"] == "Model update tonight"
    buf = io.BytesIO()
    Image.new("L", (64, 64), color=90).save(buf, "PNG")
    blocked = client.post("/predict-image", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert blocked.status_code == 503 and blocked.json()["detail"] == "Model update tonight"
    assert client.post("/predict", json=NORMAL).status_code == 200      # other features still on
    assert client.post("/admin/api/reset-stats", headers=AH).status_code == 422
    assert client.post("/admin/api/reset-stats?confirm=RESET", headers=AH).status_code == 200
    admin.settings.update(admin.DEFAULT_SETTINGS)
