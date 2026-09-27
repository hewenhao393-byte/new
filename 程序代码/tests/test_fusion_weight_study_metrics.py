from __future__ import annotations

import numpy as np
import pandas as pd

from fusion_weight_study.contracts import LABELS
from fusion_weight_study.metrics import aggregate_records, disagreement_statistics, evaluate_predictions


def _fused() -> pd.DataFrame:
    rows = []
    for record, true_index, predicted_index in [("r1", 0, 0), ("r1", 0, 1), ("r2", 1, 1)]:
        probs = np.full(6, 0.01); probs[predicted_index] = 0.95; probs /= probs.sum()
        row = {"record_id": record, "label": LABELS[true_index], "method": "equal"}
        row.update({f"prob_{label}": probs[i] for i, label in enumerate(LABELS)}); rows.append(row)
    return pd.DataFrame(rows)


def test_window_and_record_metrics_use_fixed_six_class_order() -> None:
    window_summary, window_classes, window_cm = evaluate_predictions(_fused(), "window")
    records = aggregate_records(_fused())
    record_summary, record_classes, record_cm = evaluate_predictions(records, "record")
    assert window_summary["sample_count"] == 3
    assert record_summary["sample_count"] == 2
    assert len(window_classes) == len(record_classes) == 6
    assert window_cm.shape == record_cm.shape == (6, 6)
    assert list(window_cm.index) == list(LABELS)


def test_disagreement_counts_channel_argmax_patterns() -> None:
    rows = []
    for channel, winner in ((3, 0), (4, 0), (5, 1)):
        probs = np.eye(6)[winner]
        row = {"record_id": "r1", "window_id": 0, "start_sample": 0, "end_sample": 10,
               "label": LABELS[0], "channel": channel}
        row.update({f"prob_{label}": probs[i] for i, label in enumerate(LABELS)}); rows.append(row)
    stats = disagreement_statistics(pd.DataFrame(rows), split="train_side_cv", level="window")
    assert stats.loc[0, "sample_count"] == 1
    assert stats.loc[0, "disagreement_count"] == 1
    assert stats.loc[0, "disagreement_rate"] == 1.0
    assert "CH3=" in stats.loc[0, "patterns"]
