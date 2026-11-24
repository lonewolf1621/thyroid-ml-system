from typing import Literal
import logging
import os

import pandas as pd
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import ValidationError
from pydantic import BaseModel, Field

from api.model_registry import MODEL_VERSION, load_models
from api.image_predict import image_model_loaded, model_card, predict_image
from api.demo_page import DEMO_HTML

app = FastAPI(
    title="Thyroid ML System",
    description=(
        "Hypothyroidism risk from clinical lab values (RandomForest / XGBoost / "
        "LogisticRegression pipelines) and benign vs malignant thyroid image "
        "classification (EfficientNet-B0). Research demo, not a medical device."
    ),
    version="2.0.0",
)

# -----------------------------
# Logging (no names or identifiers are collected)
# -----------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# -----------------------------
# Models
# -----------------------------
models = load_models()

# Column names the trained pipelines expect, in order.
FEATURES = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine',
    'on antithyroid medication', 'sick', 'pregnant',
    'thyroid surgery', 'I131 treatment', 'query hypothyroid',
    'query hyperthyroid', 'lithium', 'goitre', 'tumor',
    'hypopituitary', 'psych', 'TSH measured', 'TSH',
    'T3 measured', 'T3', 'TT4 measured', 'TT4',
    'T4U measured', 'T4U', 'FTI measured', 'FTI',
    'TBG measured', 'referral source',
]

# Training mapped binaryClass "P" (the dataset's negative/majority class) to 1
# and "N" (hypothyroid) to 0, so class 0 is the positive diagnosis.
HYPOTHYROID_CLASS = 0

TF = Literal["t", "f"]


class PatientData(BaseModel):
    age: float = Field(..., gt=0, lt=120)
    sex: Literal["F", "M"]

    on_thyroxine: TF = "f"
    query_on_thyroxine: TF = "f"
    on_antithyroid_medication: TF = "f"
    sick: TF = "f"
    pregnant: TF = "f"
    thyroid_surgery: TF = "f"
    I131_treatment: TF = "f"
    query_hypothyroid: TF = "f"
    query_hyperthyroid: TF = "f"
    lithium: TF = "f"
    goitre: TF = "f"
    tumor: TF = "f"
    hypopituitary: TF = "f"
    psych: TF = "f"

    TSH_measured: TF = "t"
    TSH: float = Field(..., ge=0, le=600)
    T3_measured: TF = "t"
    T3: float = Field(..., ge=0, le=20)
    TT4_measured: TF = "t"
    TT4: float = Field(..., ge=0, le=500)
    T4U_measured: TF = "t"
    T4U: float = Field(..., ge=0, le=5)
    FTI_measured: TF = "t"
    FTI: float = Field(..., ge=0, le=500)
    TBG_measured: TF = "f"
    referral_source: Literal["STMW", "SVHC", "SVHD", "SVI", "other"] = "other"

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45, "sex": "F", "TSH": 1.3, "T3": 2.0,
                "TT4": 110, "T4U": 1.0, "FTI": 110,
            }
        }
    }


def to_frame(data: PatientData) -> pd.DataFrame:
    """API field names use underscores; the pipelines expect the dataset's names."""
    row = {}
    for feature in FEATURES:
        row[feature] = getattr(data, feature.replace(" ", "_"))
    return pd.DataFrame([row], columns=FEATURES)


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return DEMO_HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "tabular_models": sorted(models),
        "image_model_loaded": image_model_loaded(),
    }


@app.get("/models")
def list_models():
    return {"version": MODEL_VERSION, "models": sorted(models), "default": "RandomForest"}


@app.get("/model-card")
def get_model_card():
    """How the image model was trained and how it scored on held-out data."""
    card = model_card()
    if not card:
        return {"architecture": "efficientnet_b0", "note": "No evaluation card for this model version."}
    return card


@app.post("/predict")
def predict(data: PatientData, model_name: str = "RandomForest"):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model not found. Available: {sorted(models)}")

    model = models[model_name]
    df = to_frame(data)

    proba = model.predict_proba(df)[0]
    classes = list(model.classes_)
    p_hypo = float(proba[classes.index(HYPOTHYROID_CLASS)])
    label = "hypothyroid" if p_hypo >= 0.5 else "negative"

    logging.info({
        "model": model_name,
        "version": MODEL_VERSION,
        "prediction": label,
        "probability_hypothyroid": p_hypo,
        "input": data.model_dump(),
    })

    return {
        "model": model_name,
        "model_version": MODEL_VERSION,
        "prediction": label,
        "probability_hypothyroid": round(p_hypo, 4),
        "disclaimer": "Research demo only. Not for clinical use.",
    }


MAX_BATCH_ROWS = 1000
MAX_BATCH_BYTES = 2 * 1024 * 1024


@app.get("/predict/batch/template", include_in_schema=True)
def batch_template():
    """A CSV with the expected columns and two example rows."""
    cols = list(PatientData.model_fields)
    rows = [{**PatientData(age=45, sex="F", TSH=1.3, T3=2.0, TT4=110, T4U=1.0, FTI=110).model_dump()},
            {**PatientData(age=55, sex="F", TSH=60, T3=0.5, TT4=40, T4U=1.0, FTI=40).model_dump()}]
    csv = pd.DataFrame(rows, columns=cols).to_csv(index=False)
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=thyroid_batch_template.csv"})


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(..., description="CSV, one patient per row (max 1000 rows)"),
                        model_name: str = "RandomForest", output: Literal["json", "csv"] = "json"):
    """Score many patients at once from a CSV.

    Columns use the API field names (see /predict/batch/template); dataset
    names with spaces are accepted too. Missing optional flags default to the
    same values as /predict. Rows that fail validation are reported, not scored.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model not found. Available: {sorted(models)}")
    data = await file.read()
    if len(data) > MAX_BATCH_BYTES:
        raise HTTPException(status_code=413, detail="CSV larger than 2 MB")
    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the file as CSV")
    if raw.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")
    if len(raw) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH_ROWS} rows per request")
    raw.columns = [c.strip().replace(" ", "_") for c in raw.columns]

    valid, results, errors = [], [], []
    for i, rec in enumerate(raw.to_dict(orient="records"), start=1):
        rec = {k: v.strip() for k, v in rec.items() if k in PatientData.model_fields and str(v).strip() != ""}
        try:
            valid.append((i, PatientData(**rec)))
        except ValidationError as e:
            errors.append({"row": i, "errors": [f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                                                for err in e.errors()]})

    if valid:
        frame = pd.concat([to_frame(p) for _, p in valid], ignore_index=True)
        model = models[model_name]
        proba = model.predict_proba(frame)[:, list(model.classes_).index(HYPOTHYROID_CLASS)]
        for (row, _), p in zip(valid, proba):
            results.append({"row": row, "prediction": "hypothyroid" if p >= 0.5 else "negative",
                            "probability_hypothyroid": round(float(p), 4)})

    logging.info({"model": model_name, "batch_rows": len(raw), "scored": len(results), "errors": len(errors)})
    if output == "csv":
        out = pd.DataFrame(results + [{"row": e["row"], "prediction": "error",
                                       "probability_hypothyroid": None, "error": "; ".join(e["errors"])}
                                      for e in errors]).sort_values("row")
        return Response(out.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=thyroid_predictions.csv"})
    return {
        "model": model_name,
        "model_version": MODEL_VERSION,
        "rows": len(raw),
        "scored": len(results),
        "hypothyroid": sum(r["prediction"] == "hypothyroid" for r in results),
        "results": results,
        "errors": errors,
        "disclaimer": "Research demo only. Not for clinical use.",
    }


@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...), explain: bool = False):
    """Benign vs malignant. Add ?explain=true for a Grad-CAM heatmap (base64 PNG)."""
    result = await predict_image(file, explain=explain)
    logging.info({"model": "efficientnet_b0", "prediction": result["prediction"],
                  "confidence": result["confidence"]})
    result["disclaimer"] = "Research demo only. Not for clinical use."
    return result
