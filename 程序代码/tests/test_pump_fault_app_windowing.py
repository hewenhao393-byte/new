from __future__ import annotations

from pathlib import Path

import numpy as np

from pump_fault_app.domain.records import PreprocessedSignalRecord
from pump_fault_app.windowing.segmenter import segment_preprocessed_signal


def _preprocessed_record(
    samples,
    *,
    sampling_rate_hz: int = 12_000,
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
    )


def test_segmenter_uses_formal_window_size_and_step() -> None:
    record = _preprocessed_record(np.arange(4800, dtype=np.float64))

    result = segment_preprocessed_signal(record)

    assert result.window_size == 2400
    assert result.step_size == 1200
    assert result.window_count == 3


def test_segmenter_produces_expected_window_boundaries() -> None:
    record = _preprocessed_record(np.arange(4800, dtype=np.float64))

    result = segment_preprocessed_signal(record)

    assert [window.window_index for window in result.windows] == [0, 1, 2]
    assert [window.start_index for window in result.windows] == [0, 1200, 2400]
    assert [window.end_index for window in result.windows] == [2400, 3600, 4800]


def test_segmenter_preserves_window_samples() -> None:
    samples = np.arange(4800, dtype=np.float64)
    record = _preprocessed_record(samples)

    result = segment_preprocessed_signal(record)

    assert np.array_equal(result.windows[0].samples, samples[0:2400])
    assert np.array_equal(result.windows[1].samples, samples[1200:3600])
    assert np.array_equal(result.windows[2].samples, samples[2400:4800])


def test_segmenter_outputs_window_time_ranges() -> None:
    record = _preprocessed_record(np.arange(4800, dtype=np.float64))

    result = segment_preprocessed_signal(record)

    assert result.windows[0].start_time_seconds == 0.0
    assert result.windows[0].end_time_seconds == 0.2
    assert result.windows[1].start_time_seconds == 0.1
    assert result.windows[1].end_time_seconds == 0.3


def test_segmenter_returns_no_windows_when_signal_shorter_than_one_window() -> None:
    record = _preprocessed_record(np.arange(1000, dtype=np.float64))

    result = segment_preprocessed_signal(record)

    assert result.window_count == 0
    assert result.windows == ()


def test_segmenter_records_overlap_ratio() -> None:
    record = _preprocessed_record(np.arange(4800, dtype=np.float64))

    result = segment_preprocessed_signal(record)

    assert result.overlap_ratio == 0.5
    assert result.window_duration_seconds == 0.2
