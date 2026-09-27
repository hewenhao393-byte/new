from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .contracts import LABELS
from .data import WINDOW_KEY
from .experiment import PROBABILITY_COLUMNS


def aggregate_records(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = [column for column in ("record_id", "parent_group_id", "label", "method", "fold", "channel") if column in frame.columns]
    return frame.groupby(grouping, as_index=False, sort=True)[list(PROBABILITY_COLUMNS)].mean()


def evaluate_predictions(frame: pd.DataFrame, level: str):
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    truth = frame["label"].to_numpy()
    macro = f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0
    )
    summary = {"level": level, "macro_f1": float(macro), "sample_count": len(frame)}
    classes = pd.DataFrame({
        "level": level, "label": LABELS, "precision": precision, "recall": recall,
        "f1": f1, "support": support.astype(int),
    })
    matrix = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=LABELS), index=LABELS, columns=LABELS
    )
    matrix.index.name = "true_label"; matrix.columns.name = "predicted_label"
    return summary, classes, matrix


def disagreement_statistics(frame: pd.DataFrame, *, split: str, level: str) -> pd.DataFrame:
    working = aggregate_records(frame) if level == "record" else frame.copy()
    keys = ["record_id"] if level == "record" else list(WINDOW_KEY)
    predictions = np.asarray(LABELS)[working.loc[:, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)]
    working = working.copy(); working["predicted_label"] = predictions
    disagree = 0; patterns = Counter()
    for _, group in working.groupby(keys, sort=True):
        channel_labels = dict(zip(group["channel"].astype(int), group["predicted_label"]))
        pattern = "|".join(f"CH{channel}={channel_labels[channel]}" for channel in sorted(channel_labels))
        patterns[pattern] += 1
        disagree += int(len(set(channel_labels.values())) > 1)
    count = sum(patterns.values())
    return pd.DataFrame([{
        "split": split, "level": level, "sample_count": count,
        "disagreement_count": disagree,
        "disagreement_rate": disagree / count if count else 0.0,
        "patterns": "; ".join(f"{key}:{value}" for key, value in sorted(patterns.items())),
    }])


def overdependence_rows(global_weights, classwise_weights) -> pd.DataFrame:
    rows = [{
        "scope": "global", "label": "ALL", "max_channel": max(global_weights, key=global_weights.get),
        "max_weight": max(global_weights.values()), "risk_flag": max(global_weights.values()) > 0.60,
    }]
    for label in LABELS:
        values = {channel: classwise_weights[channel][label] for channel in classwise_weights}
        rows.append({
            "scope": "classwise", "label": label, "max_channel": max(values, key=values.get),
            "max_weight": max(values.values()), "risk_flag": max(values.values()) > 0.60,
        })
    return pd.DataFrame(rows)
