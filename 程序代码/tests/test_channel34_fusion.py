from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_source_file_base_strips_channel_suffix():
    from pump_diagnosis.channel34_fusion import source_file_base_from_path

    path = Path(
        "/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/"
        "Motor-4/70/组合不对中4/振动_电机4_70_时域-组合不对中4-通道3.csv"
    )

    assert source_file_base_from_path(path) == "振动_电机4_70_时域-组合不对中4"


def test_align_channel_windows_keeps_only_common_keys():
    from pump_diagnosis.channel34_fusion import align_channel_windows

    ch3 = pd.DataFrame(
        [
            {"run_id": "run_a", "source_file_base": "file_a", "window_start": 0, "window_end": 4096, "label": "正常", "ch3_x": 1},
            {"run_id": "run_a", "source_file_base": "file_a", "window_start": 2048, "window_end": 6144, "label": "正常", "ch3_x": 2},
        ]
    )
    ch4 = pd.DataFrame(
        [
            {"run_id": "run_a", "source_file_base": "file_a", "window_start": 0, "window_end": 4096, "label": "正常", "ch4_y": 3},
            {"run_id": "run_b", "source_file_base": "file_b", "window_start": 0, "window_end": 4096, "label": "正常", "ch4_y": 4},
        ]
    )

    aligned_ch3, aligned_ch4, fused, unmatched = align_channel_windows(ch3, ch4)

    assert len(aligned_ch3) == 1
    assert len(aligned_ch4) == 1
    assert len(fused) == 1
    assert fused.iloc[0]["ch3_x"] == 1
    assert fused.iloc[0]["ch4_y"] == 3
    assert set(unmatched["reason"]) == {"channel3_only", "channel4_only"}


def test_build_channel34_raw_tables_prefixes_features(tmp_path, monkeypatch):
    from config import PipelineConfig
    from pump_diagnosis import channel34_fusion as fusion

    manifest = pd.DataFrame(
        [
            {
                "file_path": str(tmp_path / "a-通道4.csv"),
                "source_file_base": "a",
                "run_id": "run_a",
                "label": "正常",
                "machine_id": "Motor-4",
                "condition_id": "Motor-4_70_正常状态1",
                "rpm": 2070.0,
                "channel": 4,
                "original_fs": 20000,
                "source_column": "sig",
            }
        ]
    )
    (tmp_path / "a-通道4.csv").write_text("time,sig\n0,0\n1,1\n2,0\n3,-1\n4,0\n5,1\n6,0\n7,-1\n", encoding="utf-8")

    def fake_preprocess(samples, config):
        return samples.astype(float)

    def fake_windows(samples, window_size, step_size):
        yield 0, 4, samples[:4]

    def fake_features(window, rpm, config):
        return {
            "mean": 0.0,
            "std": 1.0,
            "rms": 1.0,
            "peak": 1.0,
            "peak_to_peak": 2.0,
            "skewness": 0.0,
            "kurtosis": 1.0,
            "crest_factor": 1.0,
            "shape_factor": 1.0,
            "impulse_factor": 1.0,
            "margin_factor": 1.0,
            "clearance_factor": 1.0,
            "variance": 1.0,
            "energy": 1.0,
            "entropy": 0.0,
            "zero_crossing_rate": 0.0,
            "waveform_length": 1.0,
            "median_abs": 1.0,
            "freq_centroid": 1.0,
            "freq_rms": 1.0,
            "freq_std": 1.0,
            "freq_skewness": 0.0,
            "freq_kurtosis": 1.0,
            "spectral_entropy": 0.0,
            "peak_frequency": 1.0,
            "peak_amplitude": 1.0,
            "band_energy_10_500": 1.0,
            "band_energy_500_1000": 1.0,
            "band_energy_1000_2000": 1.0,
            "band_energy_2000_5000": 1.0,
            "band_ratio_10_500": 0.1,
            "band_ratio_500_1000": 0.1,
            "band_ratio_1000_2000": 0.1,
            "band_ratio_2000_5000": 0.1,
            "spec_flatness": 0.1,
            "spec_rolloff_85": 1.0,
            "spec_rolloff_95": 1.0,
            "spec_flux": 0.0,
            "amp_1x": 1.0,
            "amp_2x": 1.0,
            "amp_3x": 1.0,
            "energy_1x": 1.0,
            "energy_2x": 1.0,
            "energy_3x": 1.0,
            "ratio_2x_to_1x": 1.0,
            "ratio_3x_to_1x": 1.0,
            "rotation_band_energy": 3.0,
            "wp_energy_aaa": 0.125,
            "wp_energy_aad": 0.125,
            "wp_energy_ada": 0.125,
            "wp_energy_add": 0.125,
            "wp_energy_daa": 0.125,
            "wp_energy_dad": 0.125,
            "wp_energy_dda": 0.125,
            "wp_energy_ddd": 0.125,
            "wp_entropy_aaa": 0.0,
            "wp_entropy_aad": 0.0,
            "wp_entropy_ada": 0.0,
            "wp_entropy_add": 0.0,
            "wp_entropy_daa": 0.0,
            "wp_entropy_dad": 0.0,
            "wp_entropy_dda": 0.0,
            "wp_entropy_ddd": 0.0,
            "wp_total_entropy": 0.0,
            "env_rms": 1.0,
            "env_peak": 1.0,
            "env_kurtosis": 1.0,
            "env_entropy": 0.0,
            "env_peak_frequency": 1.0,
            "env_peak_amplitude": 1.0,
            "env_band_energy_0_100": 1.0,
            "env_band_energy_100_500": 1.0,
            "env_band_energy_500_1000": 1.0,
            "env_band_energy_1000_3000": 1.0,
            "env_ratio_0_100": 0.1,
            "env_ratio_100_500": 0.1,
            "env_ratio_500_1000": 0.1,
            "env_ratio_1000_3000": 0.1,
            "env_amp_1x": 1.0,
            "env_amp_2x": 1.0,
            "env_ratio_2x_to_1x": 1.0,
            "env_spectral_entropy": 0.0,
            "env_spec_centroid": 1.0,
            "env_spec_rms": 1.0,
        }

    monkeypatch.setattr(fusion, "preprocess_run", fake_preprocess)
    monkeypatch.setattr(fusion, "iter_windows", fake_windows)
    monkeypatch.setattr(fusion, "extract_candidate_features", fake_features)

    out = tmp_path / "channel3_features_raw.csv"
    frame = fusion.extract_channel_features(manifest, "ch3", PipelineConfig(output_root=tmp_path), out)

    assert out.exists()
    assert all(column.startswith("ch3_") or column in fusion.CHANNEL34_EXTENDED_METADATA_COLUMNS for column in frame.columns)
    assert "ch3_rot_1x_amp" in frame.columns
    assert "ch3_envelope_kurtosis" in frame.columns


def test_build_balanced_training_subset_uses_even_spacing():
    from pump_diagnosis.channel34_fusion import build_balanced_training_subset

    rows = []
    for label, source_file, count in [
        ("正常", "file_a", 4),
        ("正常", "file_b", 2),
        ("松动", "file_c", 2),
    ]:
        for idx in range(count):
            rows.append(
                {
                    "split": "train",
                    "label": label,
                    "run_id": f"{source_file}_run",
                    "source_file_base": source_file,
                    "source_file": source_file,
                    "file_path": f"/tmp/{source_file}.csv",
                    "machine_id": "Motor-4",
                    "condition_id": f"{source_file}_cond",
                    "raw_fault": label,
                    "operating_condition": f"{source_file}_cond",
                    "rpm": 2070.0,
                    "speed_percent": 70,
                    "channel": 4,
                    "window_index": idx,
                    "window_start": idx * 0.1,
                    "window_end": idx * 0.1 + 0.2,
                    "original_fs": 20000,
                    "processed_fs": 12000,
                    "ch4_x": float(idx),
                }
            )
    frame = pd.DataFrame(rows)

    balanced_min, dist_min = build_balanced_training_subset(frame, "balanced_min")
    balanced_1p5, dist_1p5 = build_balanced_training_subset(frame, "balanced_1p5min")

    assert balanced_min.groupby("class_name").size().to_dict() == {"正常": 2, "松动": 2}
    assert balanced_1p5.groupby("class_name").size().to_dict() == {"正常": 3, "松动": 2}
    assert balanced_min.loc[balanced_min["source_file"] == "file_b", "window_index"].tolist() == [0]
    assert balanced_1p5.loc[balanced_1p5["source_file"] == "file_b", "window_index"].tolist() == [0]
    assert dist_min.set_index("class_name")["window_count"].to_dict() == {"正常": 2, "松动": 2}
    assert dist_1p5.set_index("class_name")["window_count"].to_dict() == {"正常": 3, "松动": 2}
