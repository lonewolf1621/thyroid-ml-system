from typing import Literal
import logging
import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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


@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    result = await predict_image(file)
    logging.info({"model": "efficientnet_b0", "prediction": result["prediction"],
                  "confidence": result["confidence"]})
    result["disclaimer"] = "Research demo only. Not for clinical use."
    return result
