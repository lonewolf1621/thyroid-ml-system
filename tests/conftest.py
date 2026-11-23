"""Shared fixtures. Tests run offline on a synthetic dataset with the same raw format as the
public data (space-separated column names, 't'/'f' flags, '?' for missing, binaryClass P/N),
so the cleaning, training and serving code paths are all exercised without a download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RAW_BOOLEAN_COLUMNS = [
    "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick", "pregnant",
    "thyroid surgery", "I131 treatment", "query hypothyroid", "query hyperthyroid", "lithium",
    "goitre", "tumor", "hypopituitary", "psych",
]


def make_raw_dataset(n: int = 800, prevalence: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sick = rng.random(n) < prevalence
    tsh = np.where(sick, rng.lognormal(3.0, 0.6, n), rng.lognormal(0.3, 0.5, n))
    tt4 = np.where(sick, rng.normal(60, 15, n), rng.normal(110, 20, n)).clip(5)
    t4u = rng.normal(0.98, 0.15, n).clip(0.3)
    raw = pd.DataFrame({
        "age": rng.integers(18, 90, n).astype(object),
        "sex": rng.choice(["F", "M", "?"], n, p=[0.65, 0.32, 0.03]),
    })
    for col in RAW_BOOLEAN_COLUMNS:
        raw[col] = rng.choice(["f", "t"], n, p=[0.92, 0.08])
    raw["query hypothyroid"] = np.where(sick & (rng.random(n) < 0.4), "t", raw["query hypothyroid"])
    for name, values in {"TSH": tsh, "T3": rng.normal(2.0, 0.5, n).clip(0.1), "TT4": tt4,
                         "T4U": t4u, "FTI": tt4 / t4u}.items():
        measured = rng.random(n) > 0.08
        raw[f"{name} measured"] = np.where(measured, "t", "f")
        raw[name] = np.where(measured, np.round(values, 2).astype(str), "?")
    raw["TBG measured"] = "f"
    raw["TBG"] = "?"
    raw["referral source"] = rng.choice(["other", "SVI", "SVHC", "STMW", "SVHD"], n)
    raw["binaryClass"] = np.where(sick, "N", "P")
    raw.loc[0, "age"] = "455"  # the real data has one impossible age
    return raw


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "hypothyroid.csv"
    make_raw_dataset().to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, raw_csv) -> str:
    from thyroid.train import train

    out = tmp_path_factory.mktemp("models")
    train(raw_csv, out)
    return str(out)


HYPOTHYROID_PATIENT = {"age": 61, "sex": "F", "TSH": 45.0, "T3": 1.0, "TT4": 48.0, "T4U": 0.9, "FTI": 53.0,
                       "query_hypothyroid": True, "referral_source": "other"}
HEALTHY_PATIENT = {"age": 34, "sex": "M", "TSH": 1.4, "T3": 2.1, "TT4": 112.0, "T4U": 1.0, "FTI": 112.0,
                   "referral_source": "SVI"}
