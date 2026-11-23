"""Input-drift detection against the training distribution.

* Numeric features: two-sample Kolmogorov–Smirnov test (drift if p < 0.01).
* Boolean and categorical features: Population Stability Index (drift if PSI > 0.2).
* The dataset is flagged as drifted when at least 30% of features drift.

CLI (exit code 1 on drift, so a scheduled CI job or cron can alert):
    python -m thyroid.drift --model-dir models --current logs/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from thyroid.schema import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, coerce_frame

KS_P_VALUE = 0.01
PSI_THRESHOLD = 0.2
DATASET_DRIFT_SHARE = 0.3
MIN_SAMPLES = 30


def psi(reference: dict[str, float], current: dict[str, float], eps: float = 1e-4) -> float:
    keys = set(reference) | set(current)
    ref = np.array([max(reference.get(k, 0.0), eps) for k in keys])
    cur = np.array([max(current.get(k, 0.0), eps) for k in keys])
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def detect_drift(reference: dict, current: pd.DataFrame) -> dict:
    current = coerce_frame(current)
    n = len(current)
    if n < MIN_SAMPLES:
        return {"status": "insufficient_data", "n_current": n, "min_samples": MIN_SAMPLES, "features": {}}

    features: dict[str, dict] = {}
    for col in NUMERIC_FEATURES:
        ref = np.asarray(reference["numeric"].get(col, []), dtype=float)
        cur = current[col].dropna().to_numpy(dtype=float)
        if len(ref) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
            features[col] = {"test": "ks", "drift": False, "note": "too few non-missing values"}
            continue
        stat, p_value = ks_2samp(ref, cur)
        features[col] = {"test": "ks", "statistic": round(float(stat), 4), "p_value": float(p_value),
                         "drift": bool(p_value < KS_P_VALUE)}

    for col in BOOLEAN_FEATURES + CATEGORICAL_FEATURES:
        cur_freq = (current[col].astype(object).where(current[col].notna(), "missing").astype(str)
                    .value_counts(normalize=True).to_dict())
        value = psi(reference["categorical"].get(col, {}), cur_freq)
        features[col] = {"test": "psi", "psi": round(value, 4), "drift": bool(value > PSI_THRESHOLD)}

    drifted = [c for c, r in features.items() if r["drift"]]
    share = len(drifted) / len(FEATURES)
    return {
        "status": "drift" if share >= DATASET_DRIFT_SHARE else "ok",
        "n_current": n,
        "drift_share": round(share, 3),
        "drifted_features": drifted,
        "thresholds": {"ks_p_value": KS_P_VALUE, "psi": PSI_THRESHOLD, "dataset_share": DATASET_DRIFT_SHARE},
        "features": features,
    }


def load_prediction_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line)["input"] for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--current", type=Path, required=True, help="Prediction log (JSONL) or CSV of recent inputs")
    parser.add_argument("--report", type=Path, help="Optional path to write the JSON report")
    args = parser.parse_args()

    reference = json.loads((args.model_dir / "drift_reference.json").read_text())
    current = pd.read_csv(args.current) if args.current.suffix == ".csv" else load_prediction_log(args.current)
    report = detect_drift(reference, current)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text)
    print(text)
    sys.exit(1 if report["status"] == "drift" else 0)


if __name__ == "__main__":
    main()
