"""Single source of truth for the model's features.

Training, the API and drift monitoring all import their feature lists from here,
so the column names used at train time and at serving time cannot drift apart.
(Version 1 of this project used ``on_thyroxine`` in the API but ``on thyroxine``
in the model, which made every prediction request fail.)
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

NUMERIC_FEATURES: list[str] = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

BOOLEAN_FEATURES: list[str] = [
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
]

CATEGORICAL_FEATURES: list[str] = ["sex", "referral_source"]

FEATURES: list[str] = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

SEX_VALUES = ("F", "M")
REFERRAL_SOURCES = ("SVHC", "SVHD", "SVI", "STMW", "other")

TARGET = "hypothyroid"  # 1 = hypothyroid (any type), 0 = negative


def normalize_name(name: str) -> str:
    """Map any spelling of a column name (spaces, hyphens, case) to a lookup key."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


CANONICAL_BY_KEY: dict[str, str] = {normalize_name(f): f for f in FEATURES}


class PatientInput(BaseModel):
    """One patient record. Lab values are optional: leave them null if not measured."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 52,
                    "sex": "F",
                    "TSH": 28.0,
                    "T3": 1.1,
                    "TT4": 58.0,
                    "T4U": 0.95,
                    "FTI": 61.0,
                    "query_hypothyroid": True,
                    "referral_source": "other",
                }
            ]
        },
    )

    age: float | None = Field(None, ge=0, le=120, description="Age in years")
    sex: Literal["F", "M"] | None = None

    TSH: float | None = Field(None, ge=0, le=600, description="Thyroid-stimulating hormone (mU/L)")
    T3: float | None = Field(None, ge=0, le=20, description="Triiodothyronine (nmol/L)")
    TT4: float | None = Field(None, ge=0, le=500, description="Total thyroxine (nmol/L)")
    T4U: float | None = Field(None, ge=0, le=5, description="Thyroxine uptake")
    FTI: float | None = Field(None, ge=0, le=500, description="Free thyroxine index")

    on_thyroxine: bool = False
    query_on_thyroxine: bool = False
    on_antithyroid_medication: bool = False
    sick: bool = False
    pregnant: bool = False
    thyroid_surgery: bool = False
    I131_treatment: bool = False
    query_hypothyroid: bool = False
    query_hyperthyroid: bool = False
    lithium: bool = False
    goitre: bool = False
    tumor: bool = False
    hypopituitary: bool = False
    psych: bool = False

    referral_source: Literal["SVHC", "SVHD", "SVI", "STMW", "other"] = "other"


def coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with exactly ``FEATURES`` in order and consistent dtypes.

    Numeric and boolean columns become float (booleans as 0.0/1.0, missing as NaN);
    categorical columns become object dtype with ``np.nan`` for missing values.
    """
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float) if col in df else np.nan
    for col in BOOLEAN_FEATURES:
        out[col] = _to_binary(df[col]) if col in df else np.nan
    for col in CATEGORICAL_FEATURES:
        if col in df:
            values = df[col].astype(object)
            out[col] = values.where(values.notna(), np.nan).astype(object)
        else:
            out[col] = pd.Series(np.nan, index=df.index, dtype=object)
    return out[FEATURES]


def _to_binary(series: pd.Series) -> pd.Series:
    mapping = {"t": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0, "y": 1.0, "yes": 1.0,
               "f": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0, "n": 0.0, "no": 0.0}
    as_text = series.astype(object).map(lambda v: str(v).strip().lower() if pd.notna(v) else None)
    return as_text.map(lambda v: mapping.get(v, np.nan) if v is not None else np.nan).astype(float)


def patients_to_frame(patients: list[PatientInput]) -> pd.DataFrame:
    """Convert validated API inputs into the model's feature frame."""
    return coerce_frame(pd.DataFrame([p.model_dump() for p in patients]))
