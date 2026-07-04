from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.motor4_group_cv import (
    LABELS,
    build_group_assignment,
    build_train_test_from_assignment,
    run_grouped_cv_for_model,
)
from pump_diagnosis.three_class_feature_table import FEATURE_COLUMNS


def test_build_group_assignment_is_group_level_and_stratified() -> None:
    rows = []
    for label in LABELS:
        for group_idx in range(5):
            for window_idx in range(3):
                rows.append(
                    {
                        "source_file": f"/tmp/{label}_{group_idx}.csv",
                        "window_id": f"{label}_{group_idx}_{window_idx}",
                        "label": label,
                    }
                )
    frame = pd.DataFrame(rows)

    assignment = build_group_assignment(frame, test_size=0.2, random_state=42)

    assert set(assignment.columns) == {"group_id", "label", "window_count", "split"}
    assert assignment["window_count"].eq(3).all()
    assert set(assignment["split"]) == {"train", "test"}
    assert assignment["group_id"].is_unique


def test_build_train_test_from_assignment_keeps_groups_disjoint() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["/tmp/a.csv"] * 4 + ["/tmp/b.csv"] * 4,
            "window_id": ["a_0_0", "a_0_1", "a_1_0", "a_1_1", "b_0_0", "b_0_1", "b_1_0", "b_1_1"],
            "label": ["正常"] * 4 + ["汽蚀"] * 4,
            **{feature: np.arange(8, dtype=float) for feature in FEATURE_COLUMNS},
        }
    )
    assignment = pd.DataFrame(
        {
            "group_id": ["/tmp/a.csv::record_0", "/tmp/a.csv::record_1", "/tmp/b.csv::record_0", "/tmp/b.csv::record_1"],
            "label": ["正常", "正常", "汽蚀", "汽蚀"],
            "window_count": [2, 2, 2, 2],
            "split": ["train", "test", "train", "test"],
        }
    )

    train, test = build_train_test_from_assignment(raw, assignment)

    assert set(train["group_id"]).isdisjoint(set(test["group_id"]))
    assert len(train) == 4
    assert len(test) == 4


def test_grouped_cv_returns_zero_group_overlap(tmp_path: Path) -> None:
    rows = []
    for label, base in [("正常", 0.0), ("转子不平衡", 2.0), ("联轴器不对中", 4.0), ("汽蚀", 6.0)]:
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
    assert isinstance(best_params, dict)
