from __future__ import annotations

from pathlib import Path

import pandas as pd

from pump_diagnosis.unified_six_class_exploration import (
    CANDIDATE_FEATURE_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    add_group_columns,
    build_consistency_report,
    build_unified_dataset,
)


def test_add_group_columns_builds_global_group_id_with_device_prefix() -> None:
    frame = pd.DataFrame(
        {
            "device_id": ["Motor-2", "Motor-4"],
            "source_file": ["/tmp/a.csv", "/tmp/b.csv"],
            "window_id": ["a_7_0", "b_3_1"],
            "label": ["正常", "汽蚀"],
            "kurtosis": [1.0, 2.0],
        }
    )

    enriched = add_group_columns(frame)

    assert enriched["record_index"].tolist() == [7, 3]
    assert enriched["group_id"].tolist() == [
        "Motor-2::/tmp/a.csv::record_7",
        "Motor-4::/tmp/b.csv::record_3",
    ]


def test_build_consistency_report_requires_matching_schema_and_processing() -> None:
    report = build_consistency_report(
        motor2_columns=["source_file", "window_id", "label", "kurtosis"],
        motor4_columns=["source_file", "window_id", "label", "kurtosis"],
        motor2_summary={"processed_fs": 12000, "window_size": 2400, "step_size": 1200, "wavelet": "db6"},
        motor4_summary={"processed_fs": 12000, "window_size": 2400, "step_size": 1200, "wavelet": "db6"},
        motor2_config={"bandpass_low": 10.0, "bandpass_high": 5000.0, "resample_up": 3, "resample_down": 5, "envelope_band_low": 2000.0, "envelope_band_high": 5000.0, "wavelet_level": 3},
        motor4_config={"bandpass_low": 10.0, "bandpass_high": 5000.0, "resample_up": 3, "resample_down": 5, "envelope_band_low": 2000.0, "envelope_band_high": 5000.0, "wavelet_level": 3},
    )

    assert bool(report["is_consistent"]) is True
    assert report["column_match"] is True
    assert report["processing_match"] is True


def test_build_unified_dataset_keeps_only_requested_candidate_features() -> None:
    motor2 = pd.DataFrame(
        {
            "source_file": ["/tmp/m2.csv"],
            "window_id": ["m2_0_0"],
            "window_start": [0],
            "window_end": [2400],
            "label": ["正常"],
            **{feature: [1.0] for feature in CANDIDATE_FEATURE_COLUMNS},
            "rms": [9.0],
        }
    )
    motor4 = pd.DataFrame(
        {
            "source_file": ["/tmp/m4.csv"],
            "window_id": ["m4_1_0"],
            "window_start": [0],
            "window_end": [2400],
            "label": ["汽蚀"],
            **{feature: [2.0] for feature in CANDIDATE_FEATURE_COLUMNS},
            "rms": [8.0],
        }
    )

    unified = build_unified_dataset(motor2, motor4)

    assert set(REQUIRED_METADATA_COLUMNS).issubset(unified.columns)
    assert set(CANDIDATE_FEATURE_COLUMNS).issubset(unified.columns)
    assert "rms" not in unified.columns
    assert unified["device_id"].tolist() == ["Motor-2", "Motor-4"]
