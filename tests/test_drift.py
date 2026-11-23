import json
from pathlib import Path

from tests.conftest import make_raw_dataset
from thyroid.data import clean
from thyroid.drift import detect_drift


def reference(model_dir):
    return json.loads((Path(model_dir) / "drift_reference.json").read_text())


def test_same_distribution_is_not_drift(model_dir):
    current, _ = clean(make_raw_dataset(400, seed=99))
    report = detect_drift(reference(model_dir), current)
    assert report["status"] == "ok"


def test_shifted_population_is_drift(model_dir):
    current, _ = clean(make_raw_dataset(400, seed=7))
    current["TSH"] *= 6
    current["TT4"] *= 0.5
    current["FTI"] *= 0.5
    current["T3"] *= 2
    current["age"] = 15
    current["sex"] = "M"
    current["pregnant"] = 1.0
    report = detect_drift(reference(model_dir), current)
    assert report["status"] == "drift"
    assert "TSH" in report["drifted_features"]


def test_small_window_reports_insufficient_data(model_dir):
    current, _ = clean(make_raw_dataset(400, seed=3))
    report = detect_drift(reference(model_dir), current.head(5))
    assert report["status"] == "insufficient_data"
