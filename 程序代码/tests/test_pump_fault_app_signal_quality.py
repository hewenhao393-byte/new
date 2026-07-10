from __future__ import annotations

from pathlib import Path

import numpy as np

from pump_fault_app.domain.records import RawSignalRecord
from pump_fault_app.quality.signal_quality import assess_signal_quality


def _record(
    samples,
    *,
    sampling_rate_hz: int = 12_000,
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


def test_good_signal_passes_quality_gate() -> None:
    time = np.arange(4800, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 50.0 * time)

    report = assess_signal_quality(_record(samples))

    assert report.allow_diagnosis is True
    assert report.quality_level == "pass"
    assert report.nan_count == 0
    assert report.inf_count == 0


def test_all_zero_signal_is_rejected() -> None:
    report = assess_signal_quality(_record(np.zeros(4800)))

    assert report.allow_diagnosis is False
    assert report.quality_level == "fail"
    assert "signal is all zeros" in report.rejection_reasons


def test_constant_signal_is_rejected() -> None:
    report = assess_signal_quality(_record(np.ones(4800) * 3.0))

    assert report.allow_diagnosis is False
    assert report.quality_level == "fail"
    assert "signal is constant" in report.rejection_reasons


def test_short_signal_is_rejected() -> None:
    report = assess_signal_quality(_record(np.arange(100.0)))

    assert report.allow_diagnosis is False
    assert report.quality_level == "fail"
    assert "signal length is shorter than one formal window" in report.rejection_reasons


def test_low_sampling_rate_is_rejected() -> None:
    report = assess_signal_quality(_record(np.arange(4800.0), sampling_rate_hz=8000))

    assert report.allow_diagnosis is False
    assert report.quality_level == "fail"
    assert "sampling rate is too low for 5000 Hz analysis" in report.rejection_reasons


def test_signal_with_nans_and_infs_is_rejected() -> None:
    samples = np.array([0.0, 1.0, np.nan, np.inf, 2.0] * 1000, dtype=np.float64)

    report = assess_signal_quality(_record(samples))

    assert report.allow_diagnosis is False
    assert report.quality_level == "fail"
    assert report.nan_count > 0
    assert report.inf_count > 0


def test_near_zero_amplitude_triggers_warning_not_failure() -> None:
    samples = np.full(4800, 1e-10, dtype=np.float64)
    samples[100] = 2e-10

    report = assess_signal_quality(_record(samples))

    assert report.quality_level == "warn"
    assert report.allow_diagnosis is True
    assert "signal amplitude is near zero" in report.warnings


def test_large_zero_ratio_triggers_warning() -> None:
    samples = np.zeros(4800, dtype=np.float64)
    samples[:1200] = np.linspace(0.1, 1.0, 1200)

    report = assess_signal_quality(_record(samples))

    assert report.quality_level == "warn"
    assert report.allow_diagnosis is True
    assert report.zero_ratio > 0.5
    assert "signal contains a large proportion of zeros" in report.warnings


def test_abrupt_jump_triggers_warning() -> None:
    samples = np.sin(np.linspace(0.0, 100.0, 4800))
    samples[2400] = 50.0

    report = assess_signal_quality(_record(samples))

    assert report.quality_level == "warn"
    assert report.allow_diagnosis is True
    assert "signal contains abrupt jumps" in report.warnings
