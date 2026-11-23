"""Train, compare and package the hypothyroidism model.

Pipeline:
  1. Stratified 80/20 split. The 20% hold-out is touched exactly once, at the end.
  2. 5-fold stratified cross-validation of each candidate on the training split.
  3. Pick the candidate with the best mean CV PR-AUC (the right metric for a ~8% positive class).
  4. Choose the decision threshold from out-of-fold predictions so that sensitivity >= 95%:
     missing a hypothyroid patient costs more than a false alarm that triggers a re-test.
  5. Refit on the full training split, evaluate on the hold-out (with bootstrap 95% CIs),
     and write the model, metadata, drift reference and a model card.

Usage:
    python -m thyroid.train --data data/hypothyroid.csv --out models [--mlflow]
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thyroid import __version__
from thyroid.data import file_sha256, load_dataset
from thyroid.schema import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

SEED = 42
TARGET_SENSITIVITY = 0.95
REFERENCE_SAMPLE_SIZE = 1000


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    boolean = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bool", boolean, BOOLEAN_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models(seed: int = SEED) -> dict[str, Pipeline]:
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=seed
        ),
    }
    return {name: Pipeline([("preprocess", build_preprocessor()), ("model", est)]) for name, est in estimators.items()}


def threshold_for_sensitivity(y_true: np.ndarray, proba: np.ndarray, target: float = TARGET_SENSITIVITY) -> float:
    """Threshold with the best Youden's J (sensitivity + specificity - 1) among those with sensitivity >= target.

    Taking simply the highest threshold that reaches the target puts the cut-off right at the
    edge of 95% sensitivity, which does not hold up on new data; maximising J above the floor
    leaves a margin whenever the classes separate well.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    ok = np.where((tpr >= target) & np.isfinite(thresholds))[0]
    if not len(ok):
        return 0.5
    best = ok[np.argmax(tpr[ok] - fpr[ok])]
    return float(np.clip(thresholds[best], 1e-6, 1 - 1e-6))


def classification_report_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    return {
        "threshold": threshold,
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "precision": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, metric, n: int = 1000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def drift_reference(X: pd.DataFrame, seed: int = SEED) -> dict:
    """Summary of the training distribution, used later to detect input drift."""
    sample = X.sample(min(REFERENCE_SAMPLE_SIZE, len(X)), random_state=seed)
    numeric = {c: [round(float(v), 4) for v in sample[c].dropna()] for c in NUMERIC_FEATURES}
    categorical = {}
    for c in BOOLEAN_FEATURES + CATEGORICAL_FEATURES:
        freqs = X[c].astype(object).where(X[c].notna(), "missing").astype(str).value_counts(normalize=True)
        categorical[c] = {str(k): round(float(v), 6) for k, v in freqs.items()}
    return {"numeric": numeric, "categorical": categorical}


def train(data_path: Path, out_dir: Path, seed: int = SEED, use_mlflow: bool = False) -> dict:
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    cv_results: dict[str, dict] = {}
    for name, pipeline in candidate_models(seed).items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "balanced_accuracy": "balanced_accuracy"},
        )
        cv_results[name] = {
            metric: {"mean": float(np.mean(scores[f"test_{metric}"])), "std": float(np.std(scores[f"test_{metric}"]))}
            for metric in ("roc_auc", "pr_auc", "balanced_accuracy")
        }
        print(f"{name:24s} CV PR-AUC {cv_results[name]['pr_auc']['mean']:.3f} ± {cv_results[name]['pr_auc']['std']:.3f}"
              f" | ROC-AUC {cv_results[name]['roc_auc']['mean']:.3f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["pr_auc"]["mean"])
    best = candidate_models(seed)[best_name]

    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = threshold_for_sensitivity(y_train.to_numpy(), oof)

    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "roc_auc_95ci": bootstrap_ci(y_true, proba, roc_auc_score),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "pr_auc_95ci": bootstrap_ci(y_true, proba, average_precision_score),
        "brier": float(brier_score_loss(y_true, proba)),
        **classification_report_at(y_true, proba, threshold),
    }

    created = datetime.now(timezone.utc)
    metadata = {
        "model_version": f"{created:%Y.%m.%d}-{file_sha256(data_path)[:8]}",
        "package_version": __version__,
        "created_utc": created.isoformat(timespec="seconds"),
        "model_type": best_name,
        "features": FEATURES,
        "target": "1 = hypothyroid (compensated, primary or secondary), 0 = negative",
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "data": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "rows_after_dedup": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "prevalence": float(y.mean()),
        },
        "cv": {"folds": 5, "seed": seed, "selection_metric": "pr_auc", "results": cv_results},
        "test": test_metrics,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "drift_reference.json").write_text(json.dumps(drift_reference(X_train, seed)))
    (out_dir / "MODEL_CARD.md").write_text(render_model_card(metadata))

    print(f"\nSelected {best_name} | threshold {threshold:.3f}")
    print(f"Hold-out ROC-AUC {test_metrics['roc_auc']:.3f} | PR-AUC {test_metrics['pr_auc']:.3f} | "
          f"sensitivity {test_metrics['sensitivity']:.3f} | specificity {test_metrics['specificity']:.3f}")

    if use_mlflow:
        log_to_mlflow(metadata, out_dir)
    return metadata


def log_to_mlflow(metadata: dict, out_dir: Path) -> None:
    import mlflow

    mlflow.set_experiment("thyroid-hypothyroidism")
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_params({
            "model_type": metadata["model_type"],
            "threshold": round(metadata["threshold"], 4),
            "target_sensitivity": metadata["target_sensitivity"],
            "seed": metadata["cv"]["seed"],
            "data_sha256": metadata["data"]["sha256"][:12],
        })
        for name, res in metadata["cv"]["results"].items():
            for metric, stats in res.items():
                mlflow.log_metric(f"cv_{name}_{metric}", stats["mean"])
        for key in ("roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "precision", "balanced_accuracy"):
            mlflow.log_metric(f"test_{key}", metadata["test"][key])
        mlflow.log_artifacts(str(out_dir), artifact_path="model_package")


def render_model_card(m: dict) -> str:
    t = m["test"]
    cm = t["confusion_matrix"]
    rows = "\n".join(
        f"| {name} | {r['roc_auc']['mean']:.3f} ± {r['roc_auc']['std']:.3f} | "
        f"{r['pr_auc']['mean']:.3f} ± {r['pr_auc']['std']:.3f} | {r['balanced_accuracy']['mean']:.3f} |"
        for name, r in m["cv"]["results"].items()
    )
    return f"""# Model card: hypothyroidism risk model

**Version:** {m['model_version']} · **Model:** {m['model_type']} · **Trained:** {m['created_utc']}

> Research and portfolio project. Not a medical device and not for clinical decisions.

## Intended use
Estimate the probability that a patient's routine thyroid panel and history indicate hypothyroidism,
to illustrate how a screening model is built, evaluated and served. Not validated for clinical use.

## Data
Public UCI / OpenML *hypothyroid* dataset (Garavan Institute, 1980s Australia), {m['data']['rows_after_dedup']:,} records
after de-duplication, {m['data']['prevalence']:.1%} positive. Target: {m['target']}.
SHA-256 of the training file: `{m['data']['sha256'][:16]}…`

## Model selection (5-fold stratified CV on the 80% training split)
| Model | ROC-AUC | PR-AUC | Balanced acc. (0.5 cut-off) |
|---|---|---|---|
{rows}

Selected by mean CV PR-AUC: **{m['model_type']}**.

## Hold-out test set ({m['data']['test_rows']} records, used once)
| Metric | Value |
|---|---|
| ROC-AUC | {t['roc_auc']:.3f} (95% CI {t['roc_auc_95ci'][0]:.3f}–{t['roc_auc_95ci'][1]:.3f}) |
| PR-AUC | {t['pr_auc']:.3f} (95% CI {t['pr_auc_95ci'][0]:.3f}–{t['pr_auc_95ci'][1]:.3f}) |
| Decision threshold | {t['threshold']:.3f} (best sensitivity–specificity balance with ≥{m['target_sensitivity']:.0%} sensitivity, from out-of-fold predictions) |
| Sensitivity (recall) | {t['sensitivity']:.3f} |
| Specificity | {t['specificity']:.3f} |
| Precision (PPV) | {t['precision']:.3f} |
| Balanced accuracy | {t['balanced_accuracy']:.3f} |
| Brier score | {t['brier']:.4f} |
| Confusion matrix | TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']} |

## Limitations
- One historical, single-centre dataset: performance on modern assays or other populations is unknown.
- Lab values are strongly predictive, so the model largely re-learns clinical reference ranges.
- Positive cases include compensated, primary and secondary hypothyroidism; the model does not distinguish them.
- No external validation and no fairness audit beyond sex being an input feature.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/hypothyroid.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mlflow", action="store_true", help="Also log the run to MLflow")
    args = parser.parse_args()
    train(args.data, args.out, args.seed, args.mlflow)


if __name__ == "__main__":
    main()
