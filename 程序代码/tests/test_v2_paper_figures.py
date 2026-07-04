from __future__ import annotations

import pandas as pd

from pump_diagnosis.v2_paper_figures import aggregate_class_distribution, select_representative_groups


def test_select_representative_groups_handles_multispeed_motor2_frame() -> None:
    rows = []
    for speed_percent, values in {50: [0.0, 1.0], 75: [2.0, 3.0]}.items():
        for record_index, feature_value in enumerate(values):
            for window_index in range(2):
                row = {
                    "device_id": "Motor-2",
                    "speed_percent": speed_percent,
                    "rpm": 740.0 if speed_percent == 50 else 1110.0,
                    "label": "正常",
                    "source_file": f"/tmp/m2_{speed_percent}_{record_index}.csv",
                    "record_index": record_index,
                    "window_id": f"{speed_percent}_{record_index}_{window_index}",
                    "group_id": f"Motor-2::{speed_percent}::{record_index}",
                    "feature_a": feature_value,
                    "feature_b": 0.0,
                }
                rows.append(row)
    frame = pd.DataFrame(rows)

    selected = select_representative_groups(frame, feature_columns=["feature_a", "feature_b"])

    assert selected["正常"]["speed_percent"] == 50
    assert selected["正常"]["record_index"] == 1


def test_aggregate_class_distribution_sums_windows_and_groups() -> None:
    frame = pd.DataFrame(
        [
            {"label": "正常", "window_count_total": 100, "group_count_total": 10},
            {"label": "正常", "window_count_total": 50, "group_count_total": 5},
            {"label": "松动", "window_count_total": 20, "group_count_total": 2},
        ]
    )

    summary = aggregate_class_distribution(frame)

    assert summary.to_dict(orient="records") == [
        {"label": "正常", "window_count_total": 150, "group_count_total": 15},
        {"label": "松动", "window_count_total": 20, "group_count_total": 2},
    ]
