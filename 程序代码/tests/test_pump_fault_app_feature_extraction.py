from __future__ import annotations

from pathlib import Path

import numpy as np

from pump_diagnosis.inference_contract import FORMAL_FEATURE_NAMES
from pump_fault_app.domain.records import PreprocessedSignalRecord
from pump_fault_app.feature_extraction.extractor import extract_formal_features
from pump_fault_app.windowing.segmenter import segment_preprocessed_signal


def _preprocessed_record(
    samples,
    *,
    sampling_rate_hz: int = 12_000,
    rpm: float = 1480.0,
) -> PreprocessedSignalRecord:
    values = np.asarray(samples, dtype=np.float64)
    return PreprocessedSignalRecord(
        source_file=Path("/tmp/demo.csv"),
        original_sample_count=int(values.size),
        processed_sample_count=int(values.size),
        original_sampling_rate_hz=sampling_rate_hz,
        target_sampling_rate_hz=sampling_rate_hz,
        was_resampled=False,
        dc_removed_before_filter=True,
        dc_removed_after_filter=True,
        filter_applied=True,
        processed_samples=values,
        processing_log=("bandpass_filter",),
        rpm=rpm,
    )


def test_segmented_windows_keep_rpm_metadata() -> None:
    record = _preprocessed_record(np.arange(4800, dtype=np.float64), rpm=2070.0)

    result = segment_preprocessed_signal(record)

    assert result.windows[0].rpm == 2070.0


def test_formal_feature_extraction_returns_frozen_feature_order() -> None:
    time = np.arange(2400, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 50.0 * time)
    record = _preprocessed_record(samples)
    window = segment_preprocessed_signal(record).windows[0]

    vector = extract_formal_features(window)

    assert vector.feature_names == FORMAL_FEATURE_NAMES
    assert len(vector.values) == 21


def test_formal_feature_extraction_produces_finite_values() -> None:
    time = np.arange(2400, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 80.0 * time) + 0.3 * np.sin(2.0 * np.pi * 160.0 * time)
    record = _preprocessed_record(samples)
    window = segment_preprocessed_signal(record).windows[0]

    vector = extract_formal_features(window)

    assert np.isfinite(np.asarray(vector.values, dtype=np.float64)).all()


def test_wavelet_energy_ratios_sum_to_one() -> None:
    time = np.arange(2400, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 100.0 * time)
    record = _preprocessed_record(samples)
    window = segment_preprocessed_signal(record).windows[0]

    vector = extract_formal_features(window)
    feature_map = dict(zip(vector.feature_names, vector.values))
    total = sum(feature_map[f"wp_energy_ratio_{index}"] for index in range(8))

    assert np.isclose(total, 1.0, atol=1e-6)


def test_rotation_ratio_features_are_non_negative() -> None:
    time = np.arange(2400, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * (1480.0 / 60.0) * time)
    record = _preprocessed_record(samples, rpm=1480.0)
    window = segment_preprocessed_signal(record).windows[0]

    vector = extract_formal_features(window)
    feature_map = dict(zip(vector.feature_names, vector.values))

    assert feature_map["rot_2x_1x_ratio"] >= 0.0
    assert feature_map["rot_3x_1x_ratio"] >= 0.0
    assert feature_map["harmonic_energy_ratio_1x_5x"] >= 0.0
