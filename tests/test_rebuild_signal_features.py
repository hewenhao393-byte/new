from __future__ import annotations

import numpy as np

from config import PipelineConfig
from pump_diagnosis.features import (
    FEATURE_COLUMNS,
    amplitude_spectrum,
    extract_candidate_features,
    wavelet_packet_features,
)
from pump_diagnosis.signal_processing import iter_windows, preprocess_run


def test_resampling_filtering_and_window_count() -> None:
    config = PipelineConfig()
    t = np.arange(240_000) / config.original_fs
    raw = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 8000 * t)

    processed = preprocess_run(raw, config)
    windows = list(iter_windows(processed, config.window_size, config.step_size))
    freqs, amps = amplitude_spectrum(processed[: config.window_size], config.processed_fs)

    assert processed.size == 144_000
    assert len(windows) == 69
    assert all(window.size == 4096 for _, _, window in windows)
    assert np.isclose(freqs[1] - freqs[0], config.processed_fs / config.window_size)
    amp_1000 = amps[np.argmin(np.abs(freqs - 1000))]
    amp_4000 = amps[np.argmin(np.abs(freqs - 4000))]
    assert amp_1000 > amp_4000 * 20


def test_candidate_feature_vector_has_84_finite_values_and_harmonics() -> None:
    config = PipelineConfig()
    t = np.arange(config.window_size) / config.processed_fs
    window = (
        1.0 * np.sin(2 * np.pi * 20 * t)
        + 0.6 * np.sin(2 * np.pi * 40 * t)
        + 0.3 * np.sin(2 * np.pi * 60 * t)
        + 0.05 * np.sin(2 * np.pi * 3000 * t)
    )

    features = extract_candidate_features(window, rpm=1200.0, config=config)

    assert list(features) == FEATURE_COLUMNS
    assert len(features) == 84
    assert np.isfinite(np.asarray(list(features.values()), dtype=float)).all()
    assert features["amp_1x"] > features["amp_2x"] > features["amp_3x"]
    assert features["energy_1x"] > features["energy_2x"] > features["energy_3x"]


def test_wavelet_packet_nodes_are_frequency_ordered_and_sum_to_one() -> None:
    config = PipelineConfig()
    t = np.arange(config.window_size) / config.processed_fs
    centers = [375, 1125, 1875, 2625, 3375, 4125, 4875, 5625]
    node_names = ["aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"]

    for expected_node, frequency in enumerate(centers):
        window = np.sin(2 * np.pi * frequency * t)
        features = wavelet_packet_features(window, config)
        ratios = np.array([features[f"wp_energy_{name}"] for name in node_names])
        assert int(np.argmax(ratios)) == expected_node
        assert np.isclose(ratios.sum(), 1.0, atol=config.wavelet_ratio_tolerance)


def test_envelope_spectrum_excludes_dc_and_detects_modulation() -> None:
    config = PipelineConfig()
    t = np.arange(config.window_size) / config.processed_fs
    window = (1.0 + 0.5 * np.sin(2 * np.pi * 50 * t)) * np.sin(2 * np.pi * 3000 * t)

    features = extract_candidate_features(window, rpm=1500.0, config=config)

    assert np.isclose(features["env_peak_frequency"], 50.0, atol=5.0)
    ratios = [
        features["env_ratio_0_100"],
        features["env_ratio_100_500"],
        features["env_ratio_500_1000"],
        features["env_ratio_1000_3000"],
    ]
    assert np.isclose(sum(ratios), 1.0, atol=1e-6)
    assert ratios[0] > ratios[1]
