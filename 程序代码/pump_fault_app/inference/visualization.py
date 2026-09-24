from __future__ import annotations

import numpy as np

from pump_fault_app.domain.formal_contract import FORMAL_V3_CONTRACT
from pump_fault_app.domain.records import (
    DiagnosisVisualizationData,
    PreprocessedSignalRecord,
    RawSignalRecord,
    SpectrumSeries,
    TimeDomainSeries,
    WaveletPacketEnergySeries,
    WindowPredictionResult,
)
from pump_fault_app.feature_extraction.extractor import _amplitude_spectrum, _envelope_signal

_DISPLAY_POINT_LIMIT = 4000


def build_diagnosis_visualization(
    *,
    raw_signal: RawSignalRecord | None,
    preprocessed_signal: PreprocessedSignalRecord | None,
    window_predictions: tuple[WindowPredictionResult, ...] | None,
) -> DiagnosisVisualizationData:
    warnings: list[str] = []
    time_domain = _safe_build(
        lambda: _build_time_domain_series(raw_signal),
        "time_domain",
        warnings,
    )
    frequency_spectrum = _safe_build(
        lambda: _build_frequency_spectrum(preprocessed_signal),
        "frequency_spectrum",
        warnings,
    )
    envelope_spectrum = _safe_build(
        lambda: _build_envelope_spectrum(preprocessed_signal),
        "envelope_spectrum",
        warnings,
    )
    wavelet_packet_energy = _safe_build(
        lambda: _build_wavelet_packet_energy(window_predictions),
        "wavelet_packet_energy",
        warnings,
    )
    return DiagnosisVisualizationData(
        time_domain=time_domain,
        frequency_spectrum=frequency_spectrum,
        envelope_spectrum=envelope_spectrum,
        wavelet_packet_energy=wavelet_packet_energy,
        warnings=tuple(warnings),
    )


def _safe_build(builder, field_name: str, warnings: list[str]):
    try:
        return builder()
    except Exception as exc:
        warnings.append(f"{field_name} visualization exploded: {exc}")
        return None


def _build_time_domain_series(raw_signal: RawSignalRecord | None) -> TimeDomainSeries | None:
    if raw_signal is None:
        return None
    samples = np.asarray(raw_signal.samples, dtype=np.float64)
    indices = _display_indices(samples.shape[0], _DISPLAY_POINT_LIMIT)
    display_samples = samples[indices]
    time_axis = indices.astype(np.float64) / float(raw_signal.sampling_rate_hz)
    return TimeDomainSeries(
        time_s=tuple(float(value) for value in time_axis),
        amplitude=tuple(float(value) for value in display_samples),
        sampling_rate_hz=raw_signal.sampling_rate_hz,
        point_count=int(display_samples.shape[0]),
        downsampled_for_display=display_samples.shape[0] < samples.shape[0],
    )


def _build_frequency_spectrum(preprocessed_signal: PreprocessedSignalRecord | None) -> SpectrumSeries | None:
    if preprocessed_signal is None:
        return None
    freqs, amps = _amplitude_spectrum(preprocessed_signal.processed_samples, preprocessed_signal.target_sampling_rate_hz)
    return _truncate_spectrum(freqs, amps)


def _build_envelope_spectrum(preprocessed_signal: PreprocessedSignalRecord | None) -> SpectrumSeries | None:
    if preprocessed_signal is None:
        return None
    envelope = _envelope_signal(preprocessed_signal.processed_samples, preprocessed_signal.target_sampling_rate_hz)
    freqs, amps = _amplitude_spectrum(envelope, preprocessed_signal.target_sampling_rate_hz)
    if amps.size:
        amps = amps.copy()
        amps[0] = 0.0
    return _truncate_spectrum(freqs, amps)


def _truncate_spectrum(freqs: np.ndarray, amps: np.ndarray) -> SpectrumSeries:
    mask = freqs <= FORMAL_V3_CONTRACT.filter_high_hz
    visible_freqs = np.asarray(freqs[mask], dtype=np.float64)
    visible_amps = np.asarray(amps[mask], dtype=np.float64)
    resolution_hz = 0.0 if visible_freqs.shape[0] < 2 else float(visible_freqs[1] - visible_freqs[0])
    return SpectrumSeries(
        frequency_hz=tuple(float(value) for value in visible_freqs),
        amplitude=tuple(float(value) for value in visible_amps),
        frequency_min_hz=0.0 if visible_freqs.size == 0 else float(visible_freqs[0]),
        frequency_max_hz=0.0 if visible_freqs.size == 0 else float(visible_freqs[-1]),
        resolution_hz=resolution_hz,
    )


def _build_wavelet_packet_energy(
    window_predictions: tuple[WindowPredictionResult, ...] | None,
) -> WaveletPacketEnergySeries | None:
    if not window_predictions:
        return None
    feature_names = window_predictions[0].feature_vector.feature_names
    ratio_names = tuple(name for name in feature_names if name.startswith("wp_energy_ratio_"))
    energy_rows = np.asarray(
        [
            [float(dict(zip(item.feature_vector.feature_names, item.feature_vector.values))[name]) for name in ratio_names]
            for item in window_predictions
        ],
        dtype=np.float64,
    )
    mean_ratios = np.mean(energy_rows, axis=0)
    total = float(np.sum(mean_ratios))
    normalized = mean_ratios if total <= 0.0 else mean_ratios / total
    band_width_hz = (FORMAL_V3_CONTRACT.target_sampling_rate / 2.0) / len(ratio_names)
    starts = tuple(float(index * band_width_hz) for index in range(len(ratio_names)))
    ends = tuple(float((index + 1) * band_width_hz) for index in range(len(ratio_names)))
    labels = tuple(_format_band_label(start, end) for start, end in zip(starts, ends))
    return WaveletPacketEnergySeries(
        band_labels=labels,
        band_start_hz=starts,
        band_end_hz=ends,
        energy_ratio=tuple(float(value) for value in normalized),
        wavelet=FORMAL_V3_CONTRACT.wavelet,
        decomposition_level=FORMAL_V3_CONTRACT.wavelet_level,
    )


def _display_indices(sample_count: int, point_limit: int) -> np.ndarray:
    if sample_count <= point_limit:
        return np.arange(sample_count, dtype=np.int64)
    return np.linspace(0, sample_count - 1, num=point_limit, dtype=np.int64)


def _format_band_label(start_hz: float, end_hz: float) -> str:
    return f"{int(round(start_hz))}-{int(round(end_hz))} Hz"
