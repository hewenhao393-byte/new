import numpy as np
import pandas as pd
import pytest

from formal_model_selection.verification import check_probability_table


def test_probability_verification_accepts_valid_six_class_rows():
    labels = [f"c{i}" for i in range(6)]
    probabilities = np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.3]])
    frame = pd.DataFrame(probabilities, columns=labels)
    frame["predicted_label"] = "c5"

    result = check_probability_table(frame, labels)

    assert result["rows"] == 1
    assert result["max_sum_error"] < 1e-12


def test_probability_verification_rejects_wrong_argmax_label():
    labels = [f"c{i}" for i in range(6)]
    frame = pd.DataFrame([[0.1, 0.2, 0.1, 0.2, 0.1, 0.3]], columns=labels)
    frame["predicted_label"] = "c0"

    with pytest.raises(AssertionError, match="argmax"):
        check_probability_table(frame, labels)
