import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import (
    CV_SPLITS,
    FEATURE_40,
    FEATURE_43,
    ITERATION_GRID,
    JOIN_KEYS,
    LABEL_ORDER,
    MAX_ITERATIONS,
    MODEL_PARAMS,
    REMOVED_FEATURES,
)
from ablation_analysis.input_validation import validate_and_merge_inputs
from baseline_analysis.config import FEATURE_COLUMNS, LABEL_ORDER as BASELINE_LABEL_ORDER
from baseline_analysis.config import MODEL_PARAMS as BASELINE_MODEL_PARAMS


def _inputs():
    keys = [
        {"record_id": "r1", "window_id": 1, "start_sample": 0, "end_sample": 4800},
        {"record_id": "r1", "window_id": 2, "start_sample": 2400, "end_sample": 7200},
    ]
    features = pd.DataFrame(keys)
    for index, feature in enumerate(FEATURE_COLUMNS):
        features[feature] = [index + 0.1, index + 0.2]
    features["label"] = ["正常", "正常"]
    features["channel"] = [4, 4]
    features["split"] = ["test", "test"]

    predictions = pd.DataFrame(keys)
    predictions["label"] = ["正常", "正常"]
    predictions["channel"] = [4, 4]
    predictions["split"] = ["test", "test"]
    predictions["predicted_label"] = ["正常", "正常"]
    return features, predictions


def test_feature_and_iteration_contracts_are_exact():
    assert FEATURE_43 == list(FEATURE_COLUMNS)
    assert len(FEATURE_43) == 43
    assert REMOVED_FEATURES == ["std", "band_energy_3000_5000_ratio", "wp_energy_ratio_7"]
    assert FEATURE_40 == [name for name in FEATURE_43 if name not in REMOVED_FEATURES]
    assert len(FEATURE_40) == 40
    assert ITERATION_GRID == list(range(20, 801, 20))
    assert MAX_ITERATIONS == 800
    assert CV_SPLITS == 5
    assert JOIN_KEYS == ["record_id", "window_id", "start_sample", "end_sample"]
    assert LABEL_ORDER is BASELINE_LABEL_ORDER
    assert MODEL_PARAMS is BASELINE_MODEL_PARAMS


def test_shuffled_predictions_join_by_keys_and_keep_prediction_order():
    features, predictions = _inputs()
    shuffled = predictions.iloc[::-1].reset_index(drop=True)

    merged = validate_and_merge_inputs(features, shuffled)

    assert merged["window_id"].tolist() == [2, 1]
    assert merged["rms"].tolist() == [0.2, 0.1]


def test_label_mismatch_is_rejected():
    features, predictions = _inputs()
    predictions.loc[0, "label"] = "松动"

    with pytest.raises(ValueError, match="label mismatch"):
        validate_and_merge_inputs(features, predictions)


@pytest.mark.parametrize("side", ["features", "predictions"])
def test_duplicate_join_keys_are_rejected_on_each_side(side):
    features, predictions = _inputs()
    if side == "features":
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    else:
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=rf"duplicate.*{side}"):
        validate_and_merge_inputs(features, predictions)


def test_prediction_without_matching_feature_is_rejected():
    features, predictions = _inputs()
    predictions.loc[0, "record_id"] = "prediction-only"

    with pytest.raises(ValueError, match="unmatched rows"):
        validate_and_merge_inputs(features, predictions)


def test_feature_rows_without_predictions_are_allowed():
    features, predictions = _inputs()
    extra = features.iloc[[0]].copy()
    extra["record_id"] = "train-only"
    extra["split"] = "train"
    features = pd.concat([features, extra], ignore_index=True)

    merged = validate_and_merge_inputs(features, predictions)

    assert len(merged) == len(predictions)
    assert merged["record_id"].tolist() == predictions["record_id"].tolist()


@pytest.mark.parametrize("side", ["features", "predictions"])
@pytest.mark.parametrize("key", JOIN_KEYS)
def test_null_join_keys_are_rejected_on_each_side(side, key):
    features, predictions = _inputs()
    target = features if side == "features" else predictions
    target.loc[0, key] = np.nan

    with pytest.raises(ValueError, match=rf"null join keys.*{side}"):
        validate_and_merge_inputs(features, predictions)


def test_feature_columns_in_predictions_are_rejected():
    features, predictions = _inputs()
    predictions[FEATURE_43[0]] = 123.0

    with pytest.raises(ValueError, match="feature columns.*predictions"):
        validate_and_merge_inputs(features, predictions)


@pytest.mark.parametrize("feature", [FEATURE_43[0], FEATURE_43[len(FEATURE_43) // 2], FEATURE_43[-1]])
def test_nonfinite_feature_value_is_rejected_across_feature_columns(feature):
    features, predictions = _inputs()
    features.loc[1, feature] = np.inf

    with pytest.raises(ValueError, match="non-finite feature values"):
        validate_and_merge_inputs(features, predictions)


@pytest.mark.parametrize("column,bad_value", [("channel", 5), ("split", "train")])
def test_optional_metadata_mismatch_is_rejected(column, bad_value):
    features, predictions = _inputs()
    predictions.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=rf"{column} mismatch"):
        validate_and_merge_inputs(features, predictions)
