from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.four_class_feature_table import (
    FourClassFeatureConfig,
    OUTPUT_COLUMNS,
    build_feature_table,
    map_fault_to_four_class,
    preprocess_signal,
)


def test_map_fault_to_four_class_filters_to_target_labels() -> None:
    assert map_fault_to_four_class("正常状态1") == "正常"
    assert map_fault_to_four_class("泵不平衡2") == "转子不平衡"
    assert map_fault_to_four_class("电机不平衡5") == "转子不平衡"
    assert map_fault_to_four_class("角向不对中3") == "联轴器不对中"
    assert map_fault_to_four_class("平行不对中2") == "联轴器不对中"
    assert map_fault_to_four_class("组合不对中4") == "联轴器不对中"
    assert map_fault_to_four_class("吸入口汽蚀1") == "汽蚀"
    assert map_fault_to_four_class("出口汽蚀5") == "汽蚀"
    assert map_fault_to_four_class("联轴器故障2D") is None
    assert map_fault_to_four_class("弯轴") is None


def test_four_class_rotation_frequency_uses_2070_rpm() -> None:
    config = FourClassFeatureConfig()

    assert np.isclose(config.rotation_frequency, 34.5)


def test_build_feature_table_keeps_only_target_four_classes(tmp_path: Path) -> None:
    data_root = tmp_path / "Vibration" / "Motor-4" / "70"
    keep_dirs = [
        data_root / "正常状态1",
        data_root / "泵不平衡1",
        data_root / "角向不对中1",
        data_root / "吸入口汽蚀1",
    ]
    skip_dir = data_root / "联轴器故障1"
    for path in keep_dirs + [skip_dir]:
        path.mkdir(parents=True)

    time = np.arange(0, 1.0, 1.0 / 1000.0)
    base_frame = pd.DataFrame(
        {
            "time": time,
            "0": np.sin(2 * np.pi * 30 * time),
            "1": np.sin(2 * np.pi * 60 * time),
        }
    )
    for path in keep_dirs:
        base_frame.to_csv(path / "x-通道4.csv", index=False)
    base_frame.to_csv(skip_dir / "y-通道4.csv", index=False)

    config = FourClassFeatureConfig(
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
    processed = preprocess_signal(base_frame["0"].to_numpy(dtype=float), config)
    expected_windows_per_run = 1 + (processed.size - config.window_size) // config.step_size

    assert list(table.columns) == OUTPUT_COLUMNS
    assert set(table["label"]) == {"正常", "转子不平衡", "联轴器不对中", "汽蚀"}
    assert table["source_file"].nunique() == 4
    assert len(table) == expected_windows_per_run * 2 * 4
