from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult, MultiChannelInferenceRequest
from pump_fault_app.domain.records import RawSignalRecord
from pump_fault_app.fusion import fuse_valid_channels
from pump_fault_app.inference.channel_inference import run_channel_inference
from pump_fault_app.io.signal_reader import RawSignalReadRequest, _load_table, read_vibration_signal
from pump_fault_app.prediction import load_catboost43_models


@dataclass(frozen=True)
class LoadedChannelSignal:
    raw_signal: RawSignalRecord
    time_axis: np.ndarray


def _explicit_time_axis(item, expected_length: int, sampling_rate_hz: int) -> np.ndarray:
    if item.time_column is None:
        return np.arange(expected_length, dtype=np.float64) / float(sampling_rate_hz)
    frame = _load_table(item.file_path)
    if frame is None or item.time_column not in frame.columns:
        raise ValueError(f"requested time column not found for {item.channel}: {item.time_column}")
    time_axis = pd.to_numeric(frame[item.time_column], errors="coerce").to_numpy(dtype=np.float64)
    if time_axis.shape != (expected_length,):
        raise ValueError(f"time and signal lengths differ for {item.channel}")
    if not np.isfinite(time_axis).all():
        raise ValueError(f"time axis contains non-finite values for {item.channel}")
    if expected_length > 1:
        intervals = np.diff(time_axis)
        if (intervals <= 0).any():
            raise ValueError(f"time axis must be strictly increasing for {item.channel}")
        expected_interval = 1.0 / float(sampling_rate_hz)
        if not np.allclose(intervals, expected_interval, rtol=0.01, atol=1e-12):
            raise ValueError(f"time intervals are incompatible with sampling rate for {item.channel}")
    return time_axis


def read_and_validate_channel_files(
    request: MultiChannelInferenceRequest,
) -> dict[str, LoadedChannelSignal]:
    time_presence = tuple(item.time_column is not None for item in request.channels)
    if any(time_presence) and not all(time_presence):
        raise ValueError("all channel files must all provide time columns or none")

    loaded: dict[str, LoadedChannelSignal] = {}
    for item in request.channels:
        raw_signal = read_vibration_signal(
            RawSignalReadRequest(
                file_path=item.file_path,
                sampling_rate_hz=request.sampling_rate_hz,
                rpm=request.rpm,
                signal_column=item.signal_column,
                time_column=item.time_column,
                measurement_position=item.channel,
            )
        )
        loaded[item.channel] = LoadedChannelSignal(
            raw_signal=raw_signal,
            time_axis=_explicit_time_axis(item, raw_signal.sample_count, request.sampling_rate_hz),
        )

    lengths = {entry.raw_signal.sample_count for entry in loaded.values()}
    if len(lengths) != 1:
        raise ValueError("channel signal lengths differ; truncation and padding are forbidden")
    reference = next(iter(loaded.values())).time_axis
    tolerance = max(1e-12, 1.0 / request.sampling_rate_hz * 1e-4)
    for channel, entry in loaded.items():
        if not np.allclose(entry.time_axis, reference, rtol=0.0, atol=tolerance):
            raise ValueError(f"channel time axes differ pointwise: {channel}")
    return loaded


def run_multichannel_inference(request: MultiChannelInferenceRequest) -> MultiChannelDiagnosisResult:
    loaded_inputs = read_and_validate_channel_files(request)
    loaded_models = load_catboost43_models(request.model_directory)
    channel_results = tuple(
        run_channel_inference(
            item,
            loaded_inputs[item.channel].raw_signal,
            request,
            loaded_models.models[item.channel],
        )
        for item in request.channels
    )
    return fuse_valid_channels(channel_results)
