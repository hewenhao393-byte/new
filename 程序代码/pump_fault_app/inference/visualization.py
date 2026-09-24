from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, hilbert, sosfiltfilt

from pump_fault_app.domain.formal_contract import FORMAL_V3_CONTRACT
from pump_fault_app.domain.records import (
    DiagnosisVisualizationData,
    PreprocessedSignalRecord,
    RawSignalRecord,
    SpectrumSeries,
    TimeDomainSeries,
    WaveletPacketEnergySeries,
)

_DISPLAY_POINT_LIMIT = 4000


def build_diagnosis_visualization(
    *,
    raw_signal: RawSignalRecord | None,
    preprocessed_signal: PreprocessedSignalRecord | None,
    window_predictions=None,
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
        lambda: _build_wavelet_packet_energy(preprocessed_signal),
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
    freqs, amps = _amplitude_spectrum(
        preprocessed_signal.processed_samples,
        preprocessed_signal.target_sampling_rate_hz,
    )
    return _truncate_spectrum(freqs, amps)


def _build_envelope_spectrum(preprocessed_signal: PreprocessedSignalRecord | None) -> SpectrumSeries | None:
    if preprocessed_signal is None:
        return None
    contract = FORMAL_V3_CONTRACT
    sos = butter(
        4,
        [contract.envelope_low_hz, contract.envelope_high_hz],
        btype="bandpass",
        fs=preprocessed_signal.target_sampling_rate_hz,
        output="sos",
    )
    envelope = np.abs(hilbert(sosfiltfilt(sos, preprocessed_signal.processed_samples)))
    envelope = envelope - np.mean(envelope)
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
    preprocessed_signal: PreprocessedSignalRecord | None,
) -> WaveletPacketEnergySeries | None:
    if preprocessed_signal is None:
        return None
    contract = FORMAL_V3_CONTRACT
    samples = np.asarray(preprocessed_signal.processed_samples, dtype=np.float64)
    rows = []
    for start in range(0, samples.size - contract.window_size + 1, contract.step_size):
        window = samples[start : start + contract.window_size]
        nodes = pywt.WaveletPacket(
            window,
            contract.wavelet,
            maxlevel=contract.wavelet_level,
        ).get_level(contract.wavelet_level, order="natural")
        energies = np.asarray([np.sum(np.square(node.data)) for node in nodes], dtype=np.float64)
        rows.append(energies / (float(np.sum(energies)) + 1e-12))
    if not rows:
        return None
    mean_ratios = np.mean(np.asarray(rows), axis=0)
    total = float(np.sum(mean_ratios))
    normalized = mean_ratios if total <= 0.0 else mean_ratios / total
    band_count = len(mean_ratios)
    band_width_hz = (FORMAL_V3_CONTRACT.target_sampling_rate / 2.0) / band_count
    starts = tuple(float(index * band_width_hz) for index in range(band_count))
    ends = tuple(float((index + 1) * band_width_hz) for index in range(band_count))
    labels = tuple(_format_band_label(start, end) for start, end in zip(starts, ends))
    return WaveletPacketEnergySeries(
        band_labels=labels,
        band_start_hz=starts,
        band_end_hz=ends,
        energy_ratio=tuple(float(value) for value in normalized),
        wavelet=FORMAL_V3_CONTRACT.wavelet,
        decomposition_level=FORMAL_V3_CONTRACT.wavelet_level,
    )


def _amplitude_spectrum(samples: np.ndarray, sampling_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    frequency = np.fft.rfftfreq(values.size, d=1.0 / float(sampling_rate_hz))
    amplitude = 2.0 * np.abs(np.fft.rfft(values * np.hanning(values.size))) / values.size
    return frequency, amplitude


def _display_indices(sample_count: int, point_limit: int) -> np.ndarray:
    if sample_count <= point_limit:
        return np.arange(sample_count, dtype=np.int64)
    return np.linspace(0, sample_count - 1, num=point_limit, dtype=np.int64)


def _format_band_label(start_hz: float, end_hz: float) -> str:
    return f"{int(round(start_hz))}-{int(round(end_hz))} Hz"
