from __future__ import annotations

import pandas as pd

from pump_diagnosis.unified_six_class_exploration_v2 import (
    CANDIDATE_FEATURE_COLUMNS,
    _attach_probability_columns,
    add_speed_metadata,
    build_unified_dataset,
    _group_vote_frame,
    _sample_svm_training_groups,
)


def test_add_speed_metadata_parses_device_speed_and_rpm() -> None:
    frame = pd.DataFrame(
        {
            "device_id": ["Motor-2", "Motor-4"],
            "source_file": [
                "/root/Vibration/Motor-2/75/正常状态1/a.csv",
                "/root/Vibration/Motor-4/70/汽蚀1/b.csv",
            ],
            "window_id": ["a_1_0", "b_2_1"],
            "label": ["正常", "汽蚀"],
        }
    )

    enriched = add_speed_metadata(frame)

    assert enriched["speed_percent"].tolist() == [75, 70]
    assert enriched["rpm"].tolist() == [1110.0, 2070.0]
    assert enriched["group_id"].tolist() == [
        "Motor-2::75::/root/Vibration/Motor-2/75/正常状态1/a.csv::record_1",
        "Motor-4::70::/root/Vibration/Motor-4/70/汽蚀1/b.csv::record_2",
    ]


def test_build_unified_dataset_v2_keeps_requested_metadata_and_features_only() -> None:
    motor2 = pd.DataFrame(
        {
            "source_file": ["/root/Vibration/Motor-2/50/正常状态1/a.csv"],
            "window_id": ["a_0_0"],
            "window_start": [0],
            "window_end": [2400],
            "label": ["正常"],
            **{feature: [1.0] for feature in CANDIDATE_FEATURE_COLUMNS},
            "rms": [9.0],
        }
    )
    motor4 = pd.DataFrame(
        {
            "source_file": ["/root/Vibration/Motor-4/70/汽蚀1/b.csv"],
            "window_id": ["b_1_0"],
            "window_start": [0],
            "window_end": [2400],
            "label": ["汽蚀"],
            **{feature: [2.0] for feature in CANDIDATE_FEATURE_COLUMNS},
            "rms": [8.0],
        }
    )

    unified = build_unified_dataset({"Motor-2-50": motor2, "Motor-4-70": motor4})

    assert set(
        ["device_id", "speed_percent", "rpm", "label", "source_file", "record_index", "window_id", "group_id"]
    ).issubset(unified.columns)
    assert set(CANDIDATE_FEATURE_COLUMNS).issubset(unified.columns)
    assert "rms" not in unified.columns


def test_group_vote_frame_can_use_device_id_as_true_column() -> None:
    frame = pd.DataFrame(
        {
            "device_id": ["Motor-2", "Motor-2", "Motor-4", "Motor-4"],
            "speed_percent": [50, 50, 70, 70],
            "rpm": [740.0, 740.0, 2070.0, 2070.0],
            "source_file": ["a.csv", "a.csv", "b.csv", "b.csv"],
            "record_index": [0, 0, 1, 1],
            "group_id": ["g1", "g1", "g2", "g2"],
            "window_id": ["w1", "w2", "w3", "w4"],
            "pred_label": ["Motor-2", "Motor-2", "Motor-4", "Motor-4"],
            "proba_Motor-2": [0.9, 0.8, 0.1, 0.2],
            "proba_Motor-4": [0.1, 0.2, 0.9, 0.8],
        }
    )

    grouped = _group_vote_frame(frame, ["Motor-2", "Motor-4"], true_column="device_id", mode="majority")

    assert grouped["true_label"].tolist() == ["Motor-2", "Motor-4"]
    assert grouped["pred_label"].tolist() == ["Motor-2", "Motor-4"]


def test_sample_svm_training_groups_keeps_groups_complete_and_covers_strata() -> None:
    rows = []
    group_specs = [
        ("g1", "Motor-2", 50, "正常"),
        ("g2", "Motor-2", 50, "正常"),
        ("g3", "Motor-2", 50, "松动"),
        ("g4", "Motor-2", 50, "松动"),
        ("g5", "Motor-2", 75, "正常"),
        ("g6", "Motor-2", 75, "正常"),
        ("g7", "Motor-4", 70, "汽蚀"),
        ("g8", "Motor-4", 70, "汽蚀"),
    ]
    for group_id, device_id, speed_percent, label in group_specs:
        for window_idx in range(2):
            rows.append(
                {
                    "group_id": group_id,
                    "device_id": device_id,
                    "speed_percent": speed_percent,
                    "rpm": 740.0 if speed_percent == 50 else 1110.0 if speed_percent == 75 else 2070.0,
                    "label": label,
                    "source_file": f"{group_id}.csv",
                    "record_index": int(group_id[1:]),
                    "window_id": f"{group_id}_{window_idx}",
                    **{feature: float(window_idx) for feature in CANDIDATE_FEATURE_COLUMNS},
                }
            )
    frame = pd.DataFrame(rows)

    sampled, report = _sample_svm_training_groups(frame, target_column="label", desired_group_count=4, random_state=42)

    assert sampled["group_id"].nunique() == 4
    assert sampled.groupby("group_id").size().tolist() == [2, 2, 2, 2]
    assert set(zip(sampled["device_id"], sampled["speed_percent"], sampled["label"])) == {
        ("Motor-2", 50, "正常"),
        ("Motor-2", 50, "松动"),
        ("Motor-2", 75, "正常"),
        ("Motor-4", 70, "汽蚀"),
    }
    assert report["group_count_selected"].sum() == 4
    assert report["window_count_selected"].sum() == 8


def test_sample_svm_training_groups_uses_device_speed_strata_for_device_task() -> None:
    rows = []
    for group_id, device_id, speed_percent in [
        ("g1", "Motor-2", 50),
        ("g2", "Motor-2", 75),
        ("g3", "Motor-4", 70),
        ("g4", "Motor-4", 70),
    ]:
        for window_idx in range(3):
            rows.append(
                {
                    "group_id": group_id,
                    "device_id": device_id,
                    "speed_percent": speed_percent,
                    "rpm": 740.0 if speed_percent == 50 else 1110.0 if speed_percent == 75 else 2070.0,
                    "label": "正常",
                    "source_file": f"{group_id}.csv",
                    "record_index": int(group_id[1:]),
                    "window_id": f"{group_id}_{window_idx}",
                    **{feature: 1.0 for feature in CANDIDATE_FEATURE_COLUMNS},
                }
            )
    frame = pd.DataFrame(rows)

    sampled, report = _sample_svm_training_groups(frame, target_column="device_id", desired_group_count=3, random_state=0)

    assert sampled["group_id"].nunique() == 3
    assert set(zip(sampled["device_id"], sampled["speed_percent"])) == {
        ("Motor-2", 50),
        ("Motor-2", 75),
        ("Motor-4", 70),
    }
    assert report["group_count_selected"].sum() == 3


def test_attach_probability_columns_respects_estimator_class_order() -> None:
    frame = pd.DataFrame({"window_id": ["w1", "w2"]})
    probability_matrix = [
        [0.2, 0.8],
        [0.9, 0.1],
    ]

    enriched = _attach_probability_columns(
        frame,
        probability_matrix=probability_matrix,
        estimator_classes=["汽蚀", "正常"],
        probability_labels=["正常", "汽蚀"],
    )

    assert enriched["proba_正常"].tolist() == [0.8, 0.1]
    assert enriched["proba_汽蚀"].tolist() == [0.2, 0.9]
