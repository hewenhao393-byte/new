"""Record-balanced diagnostics for looseness/bearing confusion errors."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_GROUPS = {
    ("松动", "松动"): "correct_looseness",
    ("松动", "轴承故障"): "looseness_to_bearing",
    ("轴承故障", "轴承故障"): "correct_bearing",
    ("轴承故障", "松动"): "bearing_to_looseness",
}

EFFECT_COMPARISONS = (
    ("correct_looseness", "looseness_to_bearing"),
    ("correct_bearing", "bearing_to_looseness"),
    ("correct_looseness", "correct_bearing"),
)

ERROR_GROUPS = ("looseness_to_bearing", "bearing_to_looseness")
METADATA_COLUMNS = ("motor", "rpm", "condition", "state", "severity")


def _assign_target_group_scalar(label, predicted_label):
    return TARGET_GROUPS.get((label, predicted_label))


def assign_target_group(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """Map actual/predicted label Series while preserving their shared index."""
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("actual and predicted must be pandas Series")
    if not actual.index.equals(predicted.index):
        raise ValueError("actual and predicted indexes must match")
    return pd.Series(
        [_assign_target_group_scalar(label, prediction) for label, prediction in zip(actual, predicted)],
        index=actual.index,
        name="target_group",
        dtype=object,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _finite_feature_values(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    _require_columns(frame, feature_columns)
    try:
        values = frame[list(feature_columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-finite or non-numeric feature values") from exc
    if not np.isfinite(values).all():
        raise ValueError("non-finite feature values")
    return values


def grouped_feature_summary(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Summarize feature distributions by target group at window level."""
    _require_columns(frame, ["target_group", "record_id"])
    _finite_feature_values(frame, feature_columns)
    rows = []
    for group, group_frame in frame.groupby("target_group", sort=False, dropna=True):
        for feature in feature_columns:
            values = group_frame[feature].astype(float)
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            rows.append(
                {
                    "target_group": group,
                    "feature": feature,
                    "window_count": len(values),
                    "record_count": group_frame["record_id"].nunique(),
                    "median": values.median(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["target_group", "feature", "window_count", "record_count", "median", "q1", "q3", "iqr"],
    )


def record_feature_medians(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Collapse overlapping windows to one median observation per record."""
    _require_columns(frame, ["target_group", "record_id"])
    _finite_feature_values(frame, feature_columns)
    return (
        frame.groupby(["target_group", "record_id"], sort=False, dropna=False)[list(feature_columns)]
        .median()
        .reset_index()
    )


def _cliffs_magnitude(absolute_delta: float) -> str:
    if absolute_delta < 0.147:
        return "negligible"
    if absolute_delta < 0.33:
        return "small"
    if absolute_delta < 0.474:
        return "medium"
    return "large"


def cliffs_delta(x: Iterable[float], y: Iterable[float]):
    """Return signed Cliff's delta, absolute delta, and magnitude label."""
    try:
        x_values = np.asarray(list(x), dtype=float)
        y_values = np.asarray(list(y), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-finite or non-numeric values") from exc
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("empty samples are not allowed")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("non-finite values are not allowed")

    comparisons = x_values[:, None] - y_values[None, :]
    delta = float((np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0)) / comparisons.size)
    absolute = abs(delta)
    return {"delta": delta, "abs_delta": absolute, "magnitude": _cliffs_magnitude(absolute)}


def effect_size_table(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Compute the three pre-specified record-level Cliff's delta comparisons."""
    record_medians = record_feature_medians(frame, feature_columns)
    rows = []
    for feature in feature_columns:
        for group_a, group_b in EFFECT_COMPARISONS:
            values_a = record_medians.loc[record_medians["target_group"].eq(group_a), feature]
            values_b = record_medians.loc[record_medians["target_group"].eq(group_b), feature]
            effect = cliffs_delta(values_a, values_b)
            rows.append(
                {
                    "feature": feature,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_records_a": len(values_a),
                    "n_records_b": len(values_b),
                    "delta": effect["delta"],
                    "abs_delta": effect["abs_delta"],
                    "magnitude": effect["magnitude"],
                    "small_sample": len(values_a) < 5 or len(values_b) < 5,
                }
            )
    return pd.DataFrame(rows)


def targeted_error_concentration(frame: pd.DataFrame):
    """Return per-record directional-error counts and top-five concentration."""
    if "target_group" not in frame.columns:
        _require_columns(frame, ["label", "predicted_label"])
        frame = frame.copy()
        frame["target_group"] = assign_target_group(frame["label"], frame["predicted_label"])
    _require_columns(frame, ["record_id", *METADATA_COLUMNS])
    errors = frame.loc[frame["target_group"].isin(ERROR_GROUPS)].copy()
    group_columns = ["target_group", "record_id", *METADATA_COLUMNS]
    per_record = (
        errors.groupby(group_columns, dropna=False, sort=False)
        .size()
        .rename("error_windows")
        .reset_index()
        .sort_values(["error_windows", "record_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    total = int(per_record["error_windows"].sum()) if not per_record.empty else 0
    top5 = int(per_record["error_windows"].head(5).sum()) if not per_record.empty else 0
    summary = {
        "total_error_windows": total,
        "error_records": len(per_record),
        "top5_error_windows": top5,
        "top5_share": top5 / total if total else 0.0,
    }
    return per_record, summary
