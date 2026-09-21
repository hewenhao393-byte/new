import numpy as np
import pandas as pd

from .config import FEATURE_43, JOIN_KEYS


def _require_columns(frame, columns, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _reject_duplicate_keys(frame, name):
    duplicate = frame.duplicated(JOIN_KEYS, keep=False)
    if duplicate.any():
        raise ValueError(f"duplicate join keys in {name}: {int(duplicate.sum())} rows")


def _reject_null_keys(frame, name):
    nulls = frame[JOIN_KEYS].isna()
    if nulls.any().any():
        columns = [column for column in JOIN_KEYS if nulls[column].any()]
        raise ValueError(f"null join keys in {name}: columns={columns}, rows={int(nulls.any(axis=1).sum())}")


def _equal_including_missing(left, right):
    return left.eq(right) | (left.isna() & right.isna())


def validate_and_merge_inputs(features, predictions):
    """Validate and attach feature columns to predictions by stable window keys."""
    _require_columns(features, JOIN_KEYS + FEATURE_43, "features")
    _require_columns(predictions, JOIN_KEYS, "predictions")
    overlapping_features = [feature for feature in FEATURE_43 if feature in predictions.columns]
    if overlapping_features:
        raise ValueError(f"feature columns are not allowed in predictions: {overlapping_features}")
    _reject_null_keys(features, "features")
    _reject_null_keys(predictions, "predictions")
    _reject_duplicate_keys(features, "features")
    _reject_duplicate_keys(predictions, "predictions")

    try:
        finite = np.isfinite(features[FEATURE_43].to_numpy(dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError("non-finite feature values or non-numeric feature columns") from exc
    if not finite.all():
        raise ValueError(f"non-finite feature values: {int((~finite).sum())} cells")

    order_column = "_prediction_row_order"
    while order_column in features.columns or order_column in predictions.columns:
        order_column = f"_{order_column}"
    ordered_predictions = predictions.assign(**{order_column: np.arange(len(predictions))})
    merged = ordered_predictions.merge(
        features,
        on=JOIN_KEYS,
        how="left",
        sort=False,
        suffixes=("", "_feature"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "unmatched rows between predictions and features: "
            f"predictions_without_features={int(merged['_merge'].eq('left_only').sum())}"
        )

    merged = merged.sort_values(order_column, kind="stable")

    for column in ("label", "channel", "split"):
        feature_column = f"{column}_feature"
        if column in predictions.columns and feature_column in merged.columns:
            matches = _equal_including_missing(merged[column], merged[feature_column])
            if not matches.all():
                raise ValueError(f"{column} mismatch between predictions and features: {int((~matches).sum())} rows")
            merged = merged.drop(columns=feature_column)

    return merged.drop(columns=["_merge", order_column]).reset_index(drop=True)
