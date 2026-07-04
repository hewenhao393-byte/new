from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.three_class_feature_table import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    ThreeClassFeatureConfig,
    build_feature_table,
    extract_window_features,
    map_fault_to_three_class,
    preprocess_signal,
)


def test_map_fault_to_three_class_filters_to_target_labels() -> None:
    assert map_fault_to_three_class("正常状态1") == "正常"
    assert map_fault_to_three_class("软脚2") == "松动"
    assert map_fault_to_three_class("电机地脚松动") == "松动"
    assert map_fault_to_three_class("轴承外圈故障3") == "轴承故障"
    assert map_fault_to_three_class("泵轴承故障1") == "轴承故障"
    assert map_fault_to_three_class("叶轮故障1") is None


def test_extract_window_features_match_requested_contract() -> None:
    config = ThreeClassFeatureConfig()
    t = np.arange(config.window_size) / config.processed_fs
    fr = config.rotation_frequency
    window = (
        1.0 * np.sin(2 * np.pi * fr * t)
        + 0.5 * np.sin(2 * np.pi * 2 * fr * t)
        + 0.2 * np.sin(2 * np.pi * 3 * fr * t)
        + 0.2 * (1.0 + 0.4 * np.sin(2 * np.pi * 80 * t)) * np.sin(2 * np.pi * 3200 * t)
    )

    features = extract_window_features(window, config)

    assert list(features) == FEATURE_COLUMNS
    assert np.isfinite(np.asarray(list(features.values()), dtype=float)).all()
    assert features["rot_1x_amp"] > features["rot_2x_amp"] > features["rot_3x_amp"]
    assert 0.0 <= features["harmonic_energy_ratio_1x_5x"] <= 1.0
    assert np.isclose(sum(features[f"wp_energy_ratio_{idx}"] for idx in range(8)), 1.0, atol=1e-6)


def test_build_feature_table_uses_channel4_and_window_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "Vibration" / "Motor-2" / "100"
    keep_dir = data_root / "正常状态1"
    skip_dir = data_root / "叶轮故障1"
    keep_dir.mkdir(parents=True)
    skip_dir.mkdir(parents=True)

    time = np.arange(0, 1.0, 1.0 / 1000.0)
    keep_frame = pd.DataFrame(
        {
            "time": time,
            "0": np.sin(2 * np.pi * 30 * time),
            "1": np.sin(2 * np.pi * 60 * time),
        }
    )
    skip_frame = pd.DataFrame({"time": time, "0": np.sin(2 * np.pi * 80 * time)})
    keep_path = keep_dir / "x-通道4.csv"
    skip_path = skip_dir / "y-通道4.csv"
    keep_frame.to_csv(keep_path, index=False)
    skip_frame.to_csv(skip_path, index=False)

    config = ThreeClassFeatureConfig(
        data_root=data_root,
        output_path=tmp_path / "features.csv",
        original_fs=1000,
        processed_fs=600,
        resample_up=3,
        resample_down=5,
        bandpass_low=10.0,
        bandpass_high=200.0,
        window_size=120,
        step_size=60,
        envelope_band_low=100.0,
        envelope_band_high=200.0,
    )

    table = build_feature_table(config)
    processed = preprocess_signal(keep_frame["0"].to_numpy(dtype=float), config)
    expected_windows_per_run = 1 + (processed.size - config.window_size) // config.step_size

    assert list(table.columns) == OUTPUT_COLUMNS
    assert set(table["label"]) == {"正常"}
    assert table["source_file"].nunique() == 1
    assert len(table) == expected_windows_per_run * 2
    assert table.iloc[0]["window_start"] == 0
    assert table.iloc[0]["window_end"] == config.window_size
    assert table["window_id"].is_unique
