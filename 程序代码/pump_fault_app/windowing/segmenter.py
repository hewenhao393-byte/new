from __future__ import annotations

import numpy as np

from pump_fault_app.domain.formal_contract import FORMAL_V3_CONTRACT
from pump_fault_app.domain.records import PreprocessedSignalRecord, SignalWindow, WindowingResult


def segment_preprocessed_signal(record: PreprocessedSignalRecord) -> WindowingResult:
    signal_contract = FORMAL_V3_CONTRACT.signal
    window_size = signal_contract.window_size
    step_size = signal_contract.step_size
    sample_rate_hz = record.target_sampling_rate_hz
    samples = np.asarray(record.processed_samples, dtype=np.float64)

    windows: list[SignalWindow] = []
    for window_index, start_index in enumerate(range(0, samples.shape[0] - window_size + 1, step_size)):
        end_index = start_index + window_size
        window_samples = samples[start_index:end_index]
        windows.append(
            SignalWindow(
                window_index=window_index,
                start_index=start_index,
                end_index=end_index,
                start_time_seconds=start_index / sample_rate_hz,
                end_time_seconds=end_index / sample_rate_hz,
                sample_rate_hz=sample_rate_hz,
                samples=window_samples,
                rpm=record.rpm,
                device_id=record.device_id,
                measurement_position=record.measurement_position,
            )
        )

    return WindowingResult(
        source_file=record.source_file,
        window_size=window_size,
        step_size=step_size,
        overlap_ratio=1.0 - (step_size / window_size),
        window_duration_seconds=window_size / sample_rate_hz,
        window_count=len(windows),
        windows=tuple(windows),
    )
