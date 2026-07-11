from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pywt
from scipy import signal, stats

from pump_fault_app.domain.formal_contract import FORMAL_V2_CONTRACT
from pump_fault_app.domain.records import FeatureVector, SampleMetadata, SignalWindow

_EPS = 1e-12


def extract_formal_features(window: SignalWindow) -> FeatureVector:
    feature_map = _extract_feature_map(window)
    ordered_values = tuple(float(feature_map[name]) for name in FORMAL_V2_CONTRACT.features.names)
    return FeatureVector(
        feature_names=FORMAL_V2_CONTRACT.features.names,
        values=ordered_values,
        metadata=SampleMetadata(
            label=None,
            rpm=window.rpm,
            device_id=window.device_id,
        ),
    )


def _extract_feature_map(window: SignalWindow) -> OrderedDict[str, float]:
    contract = FORMAL_V2_CONTRACT.signal
    samples = np.asarray(window.samples, dtype=np.float64)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    absolute_peak = float(np.max(abs_samples))
    mean_abs = float(np.mean(abs_samples))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_samples + _EPS)))

    freqs, amps = _amplitude_spectrum(samples, window.sample_rate_hz)
    power = np.square(amps)
    total_power = float(np.sum(power)) + _EPS
    spectral_weights = power / total_power

    harmonic_amps: list[float] = []
    harmonic_energies: list[float] = []
    rotation_frequency = 0.0 if window.rpm is None else float(window.rpm) / 60.0
    for multiple in (1, 2, 3):
        center = rotation_frequency * multiple
        harmonic_amps.append(_band_max_amplitude(freqs, amps, center - contract.harmonic_search_half_width_hz, center + contract.harmonic_search_half_width_hz))
        harmonic_energies.append(_band_energy(freqs, power, center - contract.harmonic_search_half_width_hz, center + contract.harmonic_search_half_width_hz))
    harmonic_energy_1x_5x = sum(
        _band_energy(
            freqs,
            power,
            rotation_frequency * multiple - contract.harmonic_search_half_width_hz,
            rotation_frequency * multiple + contract.harmonic_search_half_width_hz,
        )
        for multiple in range(1, 6)
    )

    wp = pywt.WaveletPacket(data=samples, wavelet=contract.wavelet, mode="symmetric", maxlevel=contract.wavelet_level)
    nodes = wp.get_level(contract.wavelet_level, order="freq")
    node_energies = np.asarray([float(np.sum(np.square(node.data))) for node in nodes], dtype=np.float64)
    node_ratios = node_energies / (float(np.sum(node_energies)) + _EPS)

    envelope = _envelope_signal(samples, window.sample_rate_hz)
    env_abs = np.abs(envelope)
    env_rms = float(np.sqrt(np.mean(np.square(envelope))))
    env_peak = float(np.max(env_abs))
    env_freqs, env_amps = _amplitude_spectrum(envelope, window.sample_rate_hz)
    if env_amps.size:
        env_amps = env_amps.copy()
        env_amps[0] = 0.0
    env_power = np.square(env_amps)
    env_total_power = float(np.sum(env_power)) + _EPS
    env_weights = env_power / env_total_power

    feature_map: OrderedDict[str, float] = OrderedDict()
    feature_map["kurtosis"] = float(stats.kurtosis(samples, fisher=False, bias=False))
    feature_map["skewness"] = float(stats.skew(samples, bias=False))
    feature_map["crest_factor"] = absolute_peak / (rms + _EPS)
    feature_map["impulse_factor"] = absolute_peak / (mean_abs + _EPS)
    feature_map["clearance_factor"] = absolute_peak / (sqrt_abs_mean**2 + _EPS)
    feature_map["shape_factor"] = rms / (mean_abs + _EPS)
    feature_map["rot_2x_1x_ratio"] = harmonic_amps[1] / (harmonic_amps[0] + _EPS)
    feature_map["rot_3x_1x_ratio"] = harmonic_amps[2] / (harmonic_amps[0] + _EPS)
    feature_map["harmonic_energy_ratio_1x_5x"] = harmonic_energy_1x_5x / total_power
    feature_map["spectral_entropy"] = float(-np.sum(spectral_weights * np.log(spectral_weights + _EPS)))
    feature_map["spectral_flatness"] = _spectral_flatness(power[1:])
    for index, ratio in enumerate(node_ratios):
        feature_map[f"wp_energy_ratio_{index}"] = float(ratio)
    feature_map["env_kurtosis"] = float(stats.kurtosis(envelope, fisher=False, bias=False))
    feature_map["env_crest_factor"] = env_peak / (env_rms + _EPS)

    values = np.asarray(list(feature_map.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("formal features contain non-finite values")
    return feature_map


def _amplitude_spectrum(window: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(window, dtype=np.float64)
    spectrum = np.fft.rfft(samples)
    amplitudes = np.abs(spectrum) * 2.0 / samples.shape[0]
    frequencies = np.fft.rfftfreq(samples.shape[0], d=1.0 / fs)
    return frequencies, amplitudes


def _band_mask(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    return (freqs >= low) & (freqs <= high)


def _band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = _band_mask(freqs, low, high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _band_max_amplitude(freqs: np.ndarray, amplitudes: np.ndarray, low: float, high: float) -> float:
    mask = _band_mask(freqs, low, high)
    if not np.any(mask):
        return 0.0
    return float(np.max(amplitudes[mask]))


def _spectral_flatness(power: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    valid = np.asarray(power, dtype=np.float64) + _EPS
    return float(np.exp(np.mean(np.log(valid))) / np.mean(valid))


def _envelope_signal(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    contract = FORMAL_V2_CONTRACT.signal
    sos = signal.butter(
        N=4,
        Wn=(contract.envelope_low_hz, contract.envelope_high_hz),
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    band_limited = signal.sosfiltfilt(sos, samples)
    envelope = np.abs(signal.hilbert(band_limited))
    return np.asarray(envelope - np.mean(envelope), dtype=np.float64)
