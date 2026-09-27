from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .contracts import CHANNELS, LABELS, SIMPLICITY_TOLERANCE
from .data import WINDOW_KEY
from .fusion import derive_classwise_weights, derive_global_weights, fuse_probabilities


class ExperimentError(ValueError):
    """Raised when fusion selection would violate the experiment boundary."""


PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in LABELS)


def _channel_scores(frame: pd.DataFrame) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    global_scores, class_scores = {}, {}
    for channel in CHANNELS:
        part = frame.loc[frame["channel"].eq(channel)]
        predicted = np.asarray(LABELS)[part.loc[:, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)]
        global_scores[channel] = f1_score(part["label"], predicted, labels=LABELS, average="macro", zero_division=0)
        values = f1_score(part["label"], predicted, labels=LABELS, average=None, zero_division=0)
        class_scores[channel] = dict(zip(LABELS, map(float, values)))
    return global_scores, class_scores


def derive_final_weights(oof_predictions: pd.DataFrame) -> dict[str, Mapping]:
    global_scores, class_scores = _channel_scores(oof_predictions)
    return {
        "global": derive_global_weights(global_scores),
        "classwise": derive_classwise_weights(class_scores),
    }


def channel_weight_source_metrics(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    global_scores, class_scores = _channel_scores(oof_predictions)
    rows = [
        {"channel": channel, "metric": "macro_f1", "label": "ALL", "value": global_scores[channel]}
        for channel in CHANNELS
    ]
    rows.extend(
        {"channel": channel, "metric": "class_f1", "label": label, "value": class_scores[channel][label]}
        for channel in CHANNELS for label in LABELS
    )
    return pd.DataFrame(rows)


def fuse_long_predictions(frame: pd.DataFrame, method: str, weights: Mapping | None = None) -> pd.DataFrame:
    sort_columns = list(WINDOW_KEY)
    ordered = {}
    reference = None
    metadata_columns = list(WINDOW_KEY) + ["parent_group_id", "label"]
    if "fold" in frame.columns:
        metadata_columns.append("fold")
    for channel in sorted(frame["channel"].unique()):
        part = frame.loc[frame["channel"].eq(channel)].sort_values(sort_columns).reset_index(drop=True)
        if reference is None:
            reference = part.loc[:, metadata_columns].copy()
        elif not part.loc[:, metadata_columns].astype(str).equals(reference.astype(str)):
            raise ExperimentError("channel prediction keys/labels are not aligned")
        ordered[int(channel)] = part.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    fused = fuse_probabilities(ordered, method, weights)
    result = reference.copy()
    result["method"] = method
    for index, column in enumerate(PROBABILITY_COLUMNS):
        result[column] = fused[:, index]
    return result


def cross_fitted_fusion(oof_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, provenance = [], []
    folds = sorted(int(value) for value in oof_predictions["fold"].unique())
    for outer_fold in folds:
        source = oof_predictions.loc[~oof_predictions["fold"].eq(outer_fold)]
        target = oof_predictions.loc[oof_predictions["fold"].eq(outer_fold)]
        weights = derive_final_weights(source)
        source_folds = ",".join(map(str, sorted(int(value) for value in source["fold"].unique())))
        for method in ("equal", "global", "classwise"):
            method_weights = None if method == "equal" else weights[method]
            outputs.append(fuse_long_predictions(target, method, method_weights))
            provenance.append({
                "outer_fold": outer_fold, "method": method, "source_folds": source_folds,
                "weights": method_weights,
            })
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(provenance)


def select_strategy(train_summary: pd.DataFrame, **forbidden) -> str:
    if forbidden:
        raise ExperimentError("strict results must not be passed into strategy selection")
    required = {"method", "record_macro_f1", "window_macro_f1"}
    if not required.issubset(train_summary.columns):
        raise ExperimentError("train summary lacks selection metrics")
    scores = train_summary.set_index("method")
    equal_score = float(scores.loc["equal", "record_macro_f1"])
    candidates = scores.sort_values(["record_macro_f1", "window_macro_f1"], ascending=False)
    best = str(candidates.index[0])
    if float(scores.loc[best, "record_macro_f1"]) - equal_score < SIMPLICITY_TOLERANCE:
        return "equal"
    if best == "classwise" and "global" in scores.index:
        global_score = float(scores.loc["global", "record_macro_f1"])
        if float(scores.loc[best, "record_macro_f1"]) - global_score < SIMPLICITY_TOLERANCE:
            return "global"
    return best
