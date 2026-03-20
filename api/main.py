from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field, validator
import pandas as pd
import logging
import os

from api.model_registry import load_models
from api.image_predict import predict_image

app = FastAPI(title="Thyroid ML System")

# -----------------------------
# Logging
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# -----------------------------
# Load models
# -----------------------------
models = load_models()

# -----------------------------
# Features
# -----------------------------
FEATURES = [
    'age','sex','on thyroxine','query on thyroxine',
    'on antithyroid medication','sick','pregnant',
    'thyroid surgery','I131 treatment','query hypothyroid',
    'query hyperthyroid','lithium','goitre','tumor',
    'hypopituitary','psych','TSH measured','TSH',
    'T3 measured','T3','TT4 measured','TT4',
    'T4U measured','T4U','FTI measured','FTI',
    'TBG measured','referral source'
]

# -----------------------------
# Input Schema
# -----------------------------
class PatientData(BaseModel):

    age: float = Field(..., gt=0, lt=120)

    sex: str

    on_thyroxine: str
    query_on_thyroxine: str
    on_antithyroid_medication: str
    sick: str
    pregnant: str
    thyroid_surgery: str
    I131_treatment: str

    query_hypothyroid: str
    query_hyperthyroid: str

    lithium: str
    goitre: str
    tumor: str
    hypopituitary: str
    psych: str

    TSH_measured: str
    TSH: float = Field(..., ge=0)

    T3_measured: str
    T3: float = Field(..., ge=0)

    TT4_measured: str
    TT4: float = Field(..., ge=0)

    T4U_measured: str
    T4U: float = Field(..., ge=0)

    FTI_measured: str
    FTI: float = Field(..., ge=0)

    TBG_measured: str
    referral_source: str

    @validator("*", pre=True)
    def no_empty_strings(cls, v):
        if v == "":
            raise ValueError("Empty values not allowed")
        return v

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def home():
    return {"message": "Thyroid ML API Running"}

# -----------------------------
# Tabular Prediction
# -----------------------------
@app.post("/predict")
def predict(data: PatientData, model_name: str = "RandomForest"):

    if model_name not in models:
        return {"error": "Model not found"}

    model = models[model_name]

    input_dict = data.dict()

    df = pd.DataFrame([input_dict])
    df = df[FEATURES]

    prediction = model.predict(df)[0]
    probability = float(model.predict_proba(df)[0][1])

    logging.info({
        "model": model_name,
        "prediction": int(prediction),
        "probability": probability,
        "input": input_dict
    })

    return {
        "model": model_name,
        "prediction": int(prediction),
        "probability": probability
    }

# -----------------------------
# Image Prediction
# -----------------------------
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    return await predict_image(file)