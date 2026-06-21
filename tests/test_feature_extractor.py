from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig
from pump_diagnosis.features import extract_candidate_features, wavelet_packet_features
from pump_diagnosis.labels import map_fault_to_label
from pump_diagnosis.metadata import build_file_index
from pump_diagnosis.signal_processing import preprocess_run


def test_fault_class_mapping_covers_six_target_classes() -> None:
    assert map_fault_to_label("正常状态1") == "正常"
    assert map_fault_to_label("泵不平衡3") == "转子不平衡"
    assert map_fault_to_label("角向不对中4") == "联轴器不对中"
    assert map_fault_to_label("软脚1") == "松动"
    assert map_fault_to_label("轴承外圈故障2") == "轴承故障"
    assert map_fault_to_label("吸入口汽蚀4") == "汽蚀"
    assert map_fault_to_label("正常加噪声") is None
    assert map_fault_to_label("叶轮故障1") is None


def test_preprocess_removes_dc_and_filters_high_frequency() -> None:
    config = PipelineConfig(original_fs=1000, processed_fs=1000, resample_up=1, resample_down=1, bandpass_low=10.0, bandpass_high=100.0)
    t = np.arange(0, 2, 1 / config.original_fs)
    signal = 5.0 + np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)

    processed = preprocess_run(signal, config)

    assert abs(float(np.mean(processed))) < 1e-6
    spectrum = np.abs(np.fft.rfft(processed))
    freqs = np.fft.rfftfreq(processed.size, 1 / config.processed_fs)
    amp_30 = spectrum[np.argmin(np.abs(freqs - 30))]
    amp_300 = spectrum[np.argmin(np.abs(freqs - 300))]
    assert amp_30 > amp_300 * 20


def test_wavelet_packet_features_sum_to_one() -> None:
    config = PipelineConfig(window_size=128)
    features = wavelet_packet_features(np.arange(128, dtype=float), config)
    total = sum(features[f"wp_energy_{name}"] for name in ("aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"))
    assert np.isclose(total, 1.0, atol=config.wavelet_ratio_tolerance)


def test_build_file_index_keeps_only_channel4_and_target_labels(tmp_path: Path) -> None:
    root = tmp_path / "Vibration"
    keep = root / "Motor-4" / "70" / "正常状态1" / "x-通道4.csv"
    other = root / "Motor-4" / "70" / "正常状态1" / "x-通道3.csv"
    noisy = root / "Motor-4" / "70" / "正常加噪声" / "y-通道4.csv"
    for path in (keep, other, noisy):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time": [0.00005, 0.00010], "0": [0.0, 1.0]}).to_csv(path, index=False)

    index = build_file_index(
        PipelineConfig(data_root=root, output_root=tmp_path / "output"),
        rpm_lookup={("Motor-4", 70, "正常状态1"): 2070.0},
    )

    assert index["file_path"].nunique() == 1
    assert set(index["label"]) == {"正常"}


def test_extract_candidate_features_returns_finite_values() -> None:
    config = PipelineConfig()
    t = np.arange(config.window_size) / config.processed_fs
    window = np.sin(2 * np.pi * 20 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)

    features = extract_candidate_features(window, rpm=1200.0, config=config)

    assert np.isfinite(np.asarray(list(features.values()), dtype=float)).all()
    assert features["amp_1x"] > 0
