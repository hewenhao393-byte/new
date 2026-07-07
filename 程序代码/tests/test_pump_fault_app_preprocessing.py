from __future__ import annotations

from pathlib import Path

import numpy as np

from pump_fault_app.domain.records import RawSignalRecord
from pump_fault_app.preprocessing.signal_preprocessing import preprocess_raw_signal


def _record(
    samples,
    *,
    sampling_rate_hz: int,
    rpm: float = 1480.0,
) -> RawSignalRecord:
    values = np.asarray(samples, dtype=np.float64)
    return RawSignalRecord(
        file_name="demo.csv",
        source_file=Path("/tmp/demo.csv"),
        samples=values,
        sample_count=int(values.size),
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=float(values.size / sampling_rate_hz),
        rpm=rpm,
        selected_signal_column="ch4",
    )


def test_preprocess_resamples_20000hz_signal_to_12000hz() -> None:
    fs = 20_000
    time = np.arange(fs, dtype=np.float64) / fs
    samples = np.sin(2.0 * np.pi * 50.0 * time)

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))

    assert result.target_sampling_rate_hz == 12_000
    assert result.was_resampled is True
    assert abs(result.processed_samples.shape[0] - 12_000) <= 2


def test_preprocess_does_not_resample_when_signal_already_12000hz() -> None:
    fs = 12_000
    time = np.arange(fs, dtype=np.float64) / fs
    samples = np.sin(2.0 * np.pi * 60.0 * time)

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))

    assert result.target_sampling_rate_hz == 12_000
    assert result.was_resampled is False
    assert result.original_sample_count == fs
    assert result.processed_sample_count == fs


def test_preprocess_removes_dc_offset_before_and_after_filtering() -> None:
    fs = 12_000
    time = np.arange(fs, dtype=np.float64) / fs
    samples = 3.0 + np.sin(2.0 * np.pi * 100.0 * time)

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))

    assert abs(float(np.mean(result.processed_samples))) < 1e-6
    assert result.dc_removed_before_filter is True
    assert result.dc_removed_after_filter is True


def test_preprocess_output_contains_only_finite_values() -> None:
    fs = 12_000
    time = np.arange(fs, dtype=np.float64) / fs
    samples = np.sin(2.0 * np.pi * 80.0 * time) + 0.1 * np.sin(2.0 * np.pi * 1200.0 * time)

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))

    assert np.isfinite(result.processed_samples).all()


def test_preprocess_bandpass_suppresses_very_high_frequency_component() -> None:
    fs = 20_000
    time = np.arange(fs, dtype=np.float64) / fs
    low_component = np.sin(2.0 * np.pi * 100.0 * time)
    high_component = 0.8 * np.sin(2.0 * np.pi * 7000.0 * time)
    samples = low_component + high_component

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))
    freqs = np.fft.rfftfreq(result.processed_sample_count, d=1.0 / result.target_sampling_rate_hz)
    amps = np.abs(np.fft.rfft(result.processed_samples))

    amp_100 = float(amps[int(np.argmin(np.abs(freqs - 100.0)))])
    amp_5000 = float(amps[int(np.argmin(np.abs(freqs - 5000.0)))])

    assert amp_100 > amp_5000
    assert result.filter_applied is True


def test_preprocess_records_processing_log() -> None:
    fs = 12_000
    time = np.arange(fs, dtype=np.float64) / fs
    samples = np.sin(2.0 * np.pi * 40.0 * time)

    result = preprocess_raw_signal(_record(samples, sampling_rate_hz=fs))

    assert "demean_before_resample" in result.processing_log
    assert "bandpass_filter" in result.processing_log
    assert "demean_after_filter" in result.processing_log
