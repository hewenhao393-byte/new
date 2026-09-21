import json

import numpy as np
import pandas as pd

from formal_model_selection.config import FEATURES, LABEL_ORDER
from formal_model_selection.modeling import train_final


def _tiny_six_class_table():
    rows = []
    rng = np.random.default_rng(2026)
    for split in ("train_dev", "test"):
        for label_index, label in enumerate(LABEL_ORDER):
            for record_index in range(2):
                record_id = f"{split}_{label_index}_{record_index}"
                for window_index in range(2):
                    feature_values = rng.normal(label_index, 0.1, len(FEATURES))
                    row = {
                        "record_id": record_id,
                        "window_id": window_index,
                        "split": split,
                        "label": label,
                        "motor": "Motor-2",
                        "rpm": 740,
                        "condition": label,
                        "state": str(record_index),
                        "severity": "unknown",
                        "start_sample": window_index * 2400,
                        "end_sample": window_index * 2400 + 4800,
                    }
                    row.update(dict(zip(FEATURES, feature_values)))
                    rows.append(row)
    return pd.DataFrame(rows)


def test_final_model_saves_both_window_and_record_error_sources(tmp_path):
    out = tmp_path / "run"
    params = {
        "depth": 2,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "rsm": 0.7,
    }

    train_final(_tiny_six_class_table(), "record", 3, params, 2, out)

    assert (out / "window_error_source_summary.csv").exists()
    assert (out / "record_error_source_summary.csv").exists()
    record_predictions = pd.read_csv(out / "record_predictions.csv")
    assert {
        "window_count",
        "valid_window_count",
        "mean_max_class_probability",
    } <= set(record_predictions.columns)
    assert (record_predictions["valid_window_count"] == 2).all()
    metrics = json.loads((out / "record_metrics.json").read_text(encoding="utf-8"))
    assert set(LABEL_ORDER) <= set(metrics["classification_report"])
