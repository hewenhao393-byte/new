from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PipelineConfig
from pump_diagnosis.harmonic_check import build_energy_harmonic_check, run_energy_harmonic_check


def _write_signal_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": [0.0, 0.00005, 0.00010, 0.00015],
            "0": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_csv(path, index=False)


def test_build_energy_harmonic_check_uses_workbook_rpm_for_frequencies() -> None:
    features = pd.DataFrame(
        {
            "file_path": ["/tmp/a.csv", "/tmp/a.csv"],
            "run_id": ["run_0", "run_0"],
            "sample_id": ["s0", "s1"],
            "label": ["正常", "正常"],
            "window_index": [0, 1],
            "rpm": [None, None],
            "energy_1x": [0.0, 0.0],
            "energy_2x": [0.0, 0.0],
            "energy_3x": [0.0, 0.0],
        }
    )
    file_index = pd.DataFrame(
        {
            "file_path": ["/tmp/a.csv"],
            "run_id": ["run_0"],
            "rpm": [2070.0],
        }
    )
    config = PipelineConfig()

    report = build_energy_harmonic_check(features, file_index, config)

    assert report["speed_rpm_workbook"].tolist() == [2070.0, 2070.0]
    assert report["freq_1x_hz"].tolist() == [34.5, 34.5]
    assert report["freq_2x_hz"].tolist() == [69.0, 69.0]
    assert report["freq_3x_hz"].tolist() == [103.5, 103.5]
    assert report["file_energy_1x_nunique"].tolist() == [1, 1]
    assert report["label_energy_1x_nunique"].tolist() == [1, 1]


def test_run_energy_harmonic_check_writes_csv(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    feature_root = tmp_path / "features"
    output_root = tmp_path / "output"
    workbook = tmp_path / "conditions.xlsx"
    feature_root.mkdir()

    raw_file = data_root / "Motor-4" / "70" / "正常状态1" / "振动_电机4_70_时域-正常状态1-通道4.csv"
    _write_signal_csv(raw_file)

    workbook_rows = [
        ["Measurements"],
        ["Order", "Setup", "Failure description", "Severity", "Speed (%)", "Speed (RPM) +- 5"],
        [1, "motor 4", "Healthy 1", "N/A", 0.7, 2070],
    ]
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(workbook_rows).to_excel(writer, sheet_name="Ordered Measurements", index=False, header=False)

    feature_row = {
        "sample_id": "run_0_window_0",
        "label": "正常",
        "run_id": "振动_电机4_70_时域-正常状态1-通道4_run_0",
        "file_path": str(raw_file),
        "machine_id": "Motor-4",
        "condition_id": "Motor-4_70_正常状态1",
        "rpm": None,
        "channel": 4,
        "window_index": 0,
        "window_start": 0.0,
        "window_end": 0.2,
        "original_fs": 20_000,
        "processed_fs": 12_000,
        "energy_1x": 0.0,
        "energy_2x": 0.0,
        "energy_3x": 0.0,
    }
    pd.DataFrame([feature_row]).to_csv(feature_root / "train_features_raw.csv", index=False)
    pd.DataFrame([feature_row]).to_csv(feature_root / "test_features_raw.csv", index=False)

    config = PipelineConfig(
        data_root=data_root,
        condition_workbook=workbook,
        output_root=output_root,
    )

    output = run_energy_harmonic_check(feature_root, config)

    assert (output_root / "energy_harmonic_check.csv").exists()
    assert len(output) == 2
    assert output["freq_1x_hz"].iloc[0] == 34.5
    assert output["search_band_width_hz"].iloc[0] == 10.0
