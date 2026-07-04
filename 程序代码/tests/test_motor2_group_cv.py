from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.motor2_group_cv import (
    FEATURE_COLUMNS,
    add_record_group_columns,
    build_train_test_from_assignment,
    run_grouped_cv_for_model,
)


def test_add_record_group_columns_derives_record_index_and_group_id() -> None:
    frame = pd.DataFrame(
        {
            "source_file": ["/tmp/a.csv", "/tmp/a.csv"],
            "window_id": ["file_a_7_0", "file_a_7_1"],
            "label": ["正常", "正常"],
            "rms": [0.1, 0.2],
        }
    )

    enriched = add_record_group_columns(frame)

    assert enriched["record_index"].tolist() == [7, 7]
    assert enriched["group_id"].tolist() == ["/tmp/a.csv::record_7", "/tmp/a.csv::record_7"]


def test_build_train_test_from_assignment_respects_group_boundaries() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["/tmp/a.csv"] * 4 + ["/tmp/b.csv"] * 4,
            "window_id": ["a_0_0", "a_0_1", "a_1_0", "a_1_1", "b_0_0", "b_0_1", "b_1_0", "b_1_1"],
            "label": ["正常"] * 4 + ["松动"] * 4,
            "rms": np.arange(8, dtype=float),
            "variance": np.arange(8, dtype=float) + 1.0,
        }
    )
    assignment = pd.DataFrame(
        {
            "group_id": ["/tmp/a.csv::record_0", "/tmp/a.csv::record_1", "/tmp/b.csv::record_0", "/tmp/b.csv::record_1"],
            "label": ["正常", "正常", "松动", "松动"],
            "window_count": [2, 2, 2, 2],
            "split": ["train", "test", "train", "test"],
        }
    )

    train, test = build_train_test_from_assignment(raw, assignment)

    assert set(train["group_id"]) == {"/tmp/a.csv::record_0", "/tmp/b.csv::record_0"}
    assert set(test["group_id"]) == {"/tmp/a.csv::record_1", "/tmp/b.csv::record_1"}
    assert set(train["group_id"]).isdisjoint(set(test["group_id"]))


def test_grouped_cv_uses_group_splits_and_returns_fold_scores(tmp_path: Path) -> None:
    rows = []
    for label, base in [("正常", 0.0), ("松动", 2.0), ("轴承故障", 4.0)]:
        for group_idx in range(5):
            for window_idx in range(3):
                row = {
                    "source_file": f"/tmp/{label}_{group_idx}.csv",
                    "window_id": f"{label}_{group_idx}_{window_idx}",
                    "label": label,
                    "record_index": group_idx,
                    "group_id": f"/tmp/{label}_{group_idx}.csv::record_{group_idx}",
                }
                for feature in FEATURE_COLUMNS:
                    row[feature] = base + (0.1 if feature == FEATURE_COLUMNS[0] else 0.0)
                rows.append(row)
    frame = pd.DataFrame(rows)

    cv_results, best_params = run_grouped_cv_for_model(
        "svm",
        frame,
        FEATURE_COLUMNS,
        output_root=tmp_path,
        n_splits=5,
        random_state=42,
    )

    assert not cv_results.empty
    assert cv_results["group_overlap_count"].eq(0).all()
    assert {"macro_f1", "fold", "params"}.issubset(cv_results.columns)
    assert isinstance(best_params, dict)
