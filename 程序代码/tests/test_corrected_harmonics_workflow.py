from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig


def _write_signal_csv(path: Path, frequency_hz: float = 24.6666667) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = 20_000
    t = np.arange(0, 10_000) / fs
    signal = np.sin(2 * np.pi * frequency_hz * t) + 0.05 * np.sin(2 * np.pi * 2 * frequency_hz * t)
    pd.DataFrame({"time": t, "0": signal}).to_csv(path, index=False)


def test_build_corrected_manifests_preserves_split_and_repairs_rpm(tmp_path: Path) -> None:
    data_root = tmp_path / "Vibration"
    raw_file = data_root / "Motor-2" / "100" / "正常状态1" / "振动_电机2_100_时域-正常状态1-通道4.csv"
    _write_signal_csv(raw_file)

    workbook = tmp_path / "conditions.xlsx"
    rows = [
        ["Measurements"],
        ["Order", "Setup", "Failure description", "Severity", "Speed (%)", "Speed (RPM)"],
        [1, "motor 2", "Healthy 1", "N/A", 1.0, 1480],
    ]
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Ordered Measurements", index=False, header=False)

    feature_root = tmp_path / "features"
    feature_root.mkdir()
    old_manifest = pd.DataFrame(
        {
            "file_path": [str(raw_file)],
            "label": ["正常"],
            "raw_fault": ["正常状态1"],
            "rpm": [np.nan],
            "machine_id": ["Motor-2"],
            "condition_id": ["Motor-2_100_正常状态1"],
            "run_id": ["振动_电机2_100_时域-正常状态1-通道4_run_0"],
            "source_column": ["0"],
            "channel": [4],
            "original_fs": [20_000],
            "n_samples": [10_000],
            "duration_s": [0.5],
            "status": ["indexed"],
            "sha256": ["abc"],
        }
    )
    old_manifest.to_csv(feature_root / "train_manifest.csv", index=False)
    old_manifest.to_csv(feature_root / "test_manifest.csv", index=False)

    from pump_diagnosis.corrected_harmonics import build_corrected_manifests

    config = PipelineConfig(data_root=data_root, condition_workbook=workbook, output_root=tmp_path / "output")
    train_manifest, test_manifest, report = build_corrected_manifests(feature_root, config)

    assert train_manifest["rpm"].tolist() == [1480.0]
    assert test_manifest["rpm"].tolist() == [1480.0]
    assert report["source_file_overlap_count"] == 1
    assert report["run_id_overlap_count"] == 1


def test_extract_corrected_features_adds_harmonic_aliases(tmp_path: Path) -> None:
    data_root = tmp_path / "Vibration"
    raw_file = data_root / "Motor-2" / "100" / "正常状态1" / "振动_电机2_100_时域-正常状态1-通道4.csv"
    _write_signal_csv(raw_file)

    workbook = tmp_path / "conditions.xlsx"
    rows = [
        ["Measurements"],
        ["Order", "Setup", "Failure description", "Severity", "Speed (%)", "Speed (RPM)"],
        [1, "motor 2", "Healthy 1", "N/A", 1.0, 1480],
    ]
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Ordered Measurements", index=False, header=False)

    feature_root = tmp_path / "features"
    feature_root.mkdir()
    manifest = pd.DataFrame(
        {
            "file_path": [str(raw_file)],
            "label": ["正常"],
            "raw_fault": ["正常状态1"],
            "rpm": [1480.0],
            "machine_id": ["Motor-2"],
            "condition_id": ["Motor-2_100_正常状态1"],
            "run_id": ["振动_电机2_100_时域-正常状态1-通道4_run_0"],
            "source_column": ["0"],
            "channel": [4],
            "original_fs": [20_000],
            "n_samples": [10_000],
            "duration_s": [0.5],
            "status": ["indexed"],
            "sha256": ["abc"],
        }
    )
    manifest.to_csv(feature_root / "train_manifest_corrected.csv", index=False)

    from pump_diagnosis.corrected_harmonics import extract_corrected_features, CORRECTED_FEATURE_COLUMNS

    config = PipelineConfig(data_root=data_root, condition_workbook=workbook, output_root=tmp_path / "output")
    result = extract_corrected_features(manifest, "train", config, feature_root / "train_features_raw_corrected.csv")

    assert list(result.columns)[: len(CORRECTED_FEATURE_COLUMNS) + 13]
    assert {"amp_2x_div_1x", "amp_3x_div_1x", "harmonic_energy_1x_3x"}.issubset(result.columns)
    assert np.isfinite(result[["rot_1x_amp", "rot_2x_amp", "rot_3x_amp", "energy_1x", "energy_2x", "energy_3x"]].to_numpy()).all()
