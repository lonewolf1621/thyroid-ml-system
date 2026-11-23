import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_raw_dataset
from thyroid.data import clean
from thyroid.schema import FEATURES


def test_clean_normalizes_columns_and_types():
    X, y = clean(make_raw_dataset(200))
    assert list(X.columns) == FEATURES
    assert set(y.unique()) <= {0, 1}
    assert X["on_thyroxine"].isin([0.0, 1.0]).all()
    assert X["sex"].dropna().isin(["F", "M"]).all()


def test_question_marks_become_missing_and_impossible_age_dropped():
    raw = make_raw_dataset(200)
    X, _ = clean(raw)
    assert X["TSH"].isna().any()
    assert X["age"].max() <= 120


def test_binaryclass_n_is_the_positive_class():
    raw = make_raw_dataset(300, prevalence=0.1)
    _, y = clean(raw)
    assert 0 < y.mean() < 0.5
    # The 'N' rows must be the positives: they were generated with high TSH.
    raw_x, raw_y = clean(raw)
    assert raw_x.loc[raw_y == 1, "TSH"].median() > raw_x.loc[raw_y == 0, "TSH"].median() * 3


def test_openml_class_column_supported():
    raw = make_raw_dataset(300).rename(columns=lambda c: c.replace(" ", "_"))
    raw["Class"] = np.where(raw.pop("binaryClass") == "N", "primary_hypothyroid", "negative")
    _, y = clean(raw)
    assert 0 < y.mean() < 0.5


def test_inverted_target_is_rejected():
    raw = make_raw_dataset(300)
    raw["binaryClass"] = raw["binaryClass"].map({"N": "P", "P": "N"})
    with pytest.raises(ValueError, match="inverted"):
        clean(raw)


def test_missing_feature_column_raises():
    raw = make_raw_dataset(100).drop(columns=["TSH"])
    with pytest.raises(ValueError, match="missing expected columns"):
        clean(raw)


def test_duplicates_removed():
    raw = make_raw_dataset(100)
    X, _ = clean(pd.concat([raw, raw]))
    X_single, _ = clean(raw)
    assert len(X) == len(X_single)
