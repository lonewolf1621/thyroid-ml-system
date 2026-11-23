import numpy as np
import pytest
from pydantic import ValidationError

from tests.conftest import HYPOTHYROID_PATIENT
from thyroid.schema import FEATURES, PatientInput, patients_to_frame


def test_frame_has_model_columns_in_order():
    frame = patients_to_frame([PatientInput(**HYPOTHYROID_PATIENT)])
    assert list(frame.columns) == FEATURES
    assert frame.loc[0, "query_hypothyroid"] == 1.0
    assert frame.loc[0, "on_thyroxine"] == 0.0


def test_empty_patient_is_valid_and_missing_labs_are_nan():
    frame = patients_to_frame([PatientInput()])
    assert np.isnan(frame.loc[0, "TSH"])
    assert frame.loc[0, "referral_source"] == "other"


@pytest.mark.parametrize("bad", [
    {"age": -1},
    {"age": 200},
    {"TSH": -0.5},
    {"sex": "X"},
    {"referral_source": "hospital"},
    {"on thyroxine": True},  # v1 spelling must be rejected, not silently ignored
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValidationError):
        PatientInput(**bad)
