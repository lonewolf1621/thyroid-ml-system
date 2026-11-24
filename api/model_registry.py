import os
from pathlib import Path

import joblib

MODEL_VERSION = "v1"

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / MODEL_VERSION


def load_models():
    """Load every .pkl pipeline in models/<version>/ keyed by file name."""
    models = {}
    for file in sorted(os.listdir(MODEL_DIR)):
        if not file.endswith(".pkl"):
            continue
        name = file[: -len(".pkl")]
        try:
            models[name] = joblib.load(MODEL_DIR / file)
        except Exception as e:  # e.g. a missing optional library
            print(f"[models] Could not load {file}: {e}")
    return models
