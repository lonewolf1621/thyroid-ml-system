"""Load, download and clean the UCI / OpenML hypothyroid dataset.

Usage:
    python -m thyroid.data download --out data/hypothyroid.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from thyroid.schema import CANONICAL_BY_KEY, FEATURES, TARGET, coerce_frame, normalize_name

OPENML_DATA_ID = 57  # "hypothyroid" (Garavan Institute / UCI), 3,772 records
DEFAULT_PATH = Path("data/hypothyroid.csv")


def download(out: Path = DEFAULT_PATH, force: bool = False) -> Path:
    """Download the public dataset from OpenML and save it as CSV (skips if present)."""
    out = Path(out)
    if out.exists() and not force:
        print(f"Dataset already present: {out}")
        return out
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    frame = bunch.frame
    if len(frame) < 3000:
        raise RuntimeError(f"Unexpected dataset size {len(frame)}; is OpenML id {OPENML_DATA_ID} still hypothyroid?")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame)} rows to {out}")
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: Path = DEFAULT_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read a raw CSV and return (X, y) with canonical features and a 0/1 target."""
    raw = pd.read_csv(path, na_values=["?", "", " "], keep_default_na=True)
    return clean(raw)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize column names, build the target, fix obvious data errors, drop duplicates."""
    df = raw.copy()
    df.columns = [CANONICAL_BY_KEY.get(normalize_name(c), normalize_name(c)) for c in df.columns]

    y = _build_target(df)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    X = coerce_frame(df)
    # One record in the source data has age 455; treat impossible ages as missing.
    X.loc[(X["age"] <= 0) | (X["age"] > 120), "age"] = np.nan
    X.loc[~X["sex"].isin(["F", "M"]), "sex"] = np.nan

    combined = X.assign(**{TARGET: y.values}).drop_duplicates()
    X = combined[FEATURES].reset_index(drop=True)
    y = combined[TARGET].astype(int).reset_index(drop=True).rename(TARGET)
    return X, y


def _build_target(df: pd.DataFrame) -> pd.Series:
    """1 = hypothyroid, 0 = negative.

    * OpenML / UCI ``Class``: negative | compensated_hypothyroid | primary_hypothyroid |
      secondary_hypothyroid -> anything other than "negative" is a positive case.
    * Kaggle ``binaryClass``: "N" marks the 291 hypothyroid cases and "P" the 3,481
      negatives. (Version 1 mapped P -> 1, so its "probability" was the probability
      of being healthy.)
    """
    if "class" in df.columns:
        labels = df["class"].astype(str).str.strip().str.lower()
        y = (labels != "negative").astype(int)
    elif "binaryclass" in df.columns:
        labels = df["binaryclass"].astype(str).str.strip().str.upper()
        if not set(labels.unique()) <= {"P", "N"}:
            raise ValueError(f"Unexpected binaryClass values: {sorted(labels.unique())}")
        y = (labels == "N").astype(int)
    elif TARGET in df.columns:
        y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    else:
        raise ValueError("No target column found (expected 'Class', 'binaryClass' or 'hypothyroid').")

    prevalence = float(y.mean())
    if not 0 < prevalence < 0.5:
        raise ValueError(
            f"Positive-class prevalence is {prevalence:.1%}. Hypothyroidism should be the minority "
            "class; the target mapping is probably inverted."
        )
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Download the public dataset from OpenML")
    dl.add_argument("--out", type=Path, default=DEFAULT_PATH)
    dl.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        download(args.out, args.force)


if __name__ == "__main__":
    main()
