from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pywt
from scipy import signal, stats

from config import PipelineConfig


TIME_FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "margin_factor",
    "clearance_factor",
    "variance",
    "energy",
    "entropy",
    "zero_crossing_rate",
    "waveform_length",
    "median_abs",
]

FREQUENCY_FEATURES = [
    "freq_centroid",
    "freq_rms",
    "freq_std",
    "freq_skewness",
    "freq_kurtosis",
    "spectral_entropy",
    "peak_frequency",
    "peak_amplitude",
    "band_energy_10_500",
    "band_energy_500_1000",
    "band_energy_1000_2000",
    "band_energy_2000_5000",
    "band_ratio_10_500",
    "band_ratio_500_1000",
    "band_ratio_1000_2000",
    "band_ratio_2000_5000",
    "spec_flatness",
    "spec_rolloff_85",
    "spec_rolloff_95",
    "spec_flux",
]

ROTATION_FEATURES = [
    "amp_1x",
    "amp_2x",
    "amp_3x",
    "energy_1x",
    "energy_2x",
    "energy_3x",
    "ratio_2x_to_1x",
    "ratio_3x_to_1x",
    "rotation_band_energy",
]

_WAVELET_NODES = ["aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"]
WAVELET_FEATURES = [f"wp_energy_{node}" for node in _WAVELET_NODES] + [
    f"wp_entropy_{node}" for node in _WAVELET_NODES
] + ["wp_total_entropy"]

ENVELOPE_FEATURES = [
    "env_rms",
    "env_peak",
    "env_kurtosis",
    "env_entropy",
    "env_peak_frequency",
    "env_peak_amplitude",
    "env_band_energy_0_100",
    "env_band_energy_100_500",
    "env_band_energy_500_1000",
    "env_band_energy_1000_3000",
    "env_ratio_0_100",
    "env_ratio_100_500",
    "env_ratio_500_1000",
    "env_ratio_1000_3000",
    "env_amp_1x",
    "env_amp_2x",
    "env_ratio_2x_to_1x",
    "env_spectral_entropy",
    "env_spec_centroid",
    "env_spec_rms",
]

FEATURE_COLUMNS = TIME_FEATURES + FREQUENCY_FEATURES + ROTATION_FEATURES + WAVELET_FEATURES + ENVELOPE_FEATURES

_EPS = 1e-12


def amplitude_spectrum(window: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(window, dtype=np.float64)
    spectrum = np.fft.rfft(samples)
    amplitudes = np.abs(spectrum) * 2.0 / samples.shape[0]
    frequencies = np.fft.rfftfreq(samples.shape[0], d=1.0 / fs)
    return frequencies, amplitudes


def extract_candidate_features(
    window: np.ndarray,
    *,
    rpm: float | None,
    config: PipelineConfig,
) -> OrderedDict[str, float]:
    features: dict[str, float] = {}
    features.update(_time_features(window))
    features.update(_frequency_features(window, config))
    features.update(_rotation_features(window, rpm, config))
    features.update(wavelet_packet_features(window, config))
    features.update(_envelope_features(window, rpm, config))
    ordered = OrderedDict((name, float(features[name])) for name in FEATURE_COLUMNS)
    values = np.asarray(list(ordered.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("candidate features contain non-finite values")
    return ordered


def wavelet_packet_features(window: np.ndarray, config: PipelineConfig) -> dict[str, float]:
    wp = pywt.WaveletPacket(
        data=np.asarray(window, dtype=np.float64),
        wavelet=config.wavelet,
        mode=config.wavelet_mode,
        maxlevel=config.wavelet_level,
    )
    nodes = wp.get_level(config.wavelet_level, order="freq")
    energies = np.asarray([float(np.sum(np.square(node.data))) for node in nodes], dtype=np.float64)
    total_energy = float(np.sum(energies)) + _EPS
    ratios = energies / total_energy
    features: dict[str, float] = {}
    entropies: list[float] = []
    for node_name, ratio, node in zip(_WAVELET_NODES, ratios, nodes):
        normalized = np.square(np.asarray(node.data, dtype=np.float64))
        normalized = normalized / (float(np.sum(normalized)) + _EPS)
        entropy_value = float(-np.sum(normalized * np.log(normalized + _EPS)))
        features[f"wp_energy_{node_name}"] = float(ratio)
        features[f"wp_entropy_{node_name}"] = entropy_value
        entropies.append(entropy_value)
    features["wp_total_entropy"] = float(np.sum(entropies))
    return features


def _time_features(window: np.ndarray) -> dict[str, float]:
    samples = np.asarray(window, dtype=np.float64)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(abs_samples))
    mean_abs = float(np.mean(abs_samples))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_samples + _EPS)))
    probabilities = np.square(samples) / (float(np.sum(np.square(samples))) + _EPS)
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(samples)),
        "skewness": float(stats.skew(samples, bias=False)),
        "kurtosis": float(stats.kurtosis(samples, fisher=False, bias=False)),
        "crest_factor": peak / (rms + _EPS),
        "shape_factor": rms / (mean_abs + _EPS),
        "impulse_factor": peak / (mean_abs + _EPS),
        "margin_factor": peak / (sqrt_abs_mean**2 + _EPS),
        "clearance_factor": peak / (np.mean(np.sqrt(abs_samples + _EPS)) ** 2 + _EPS),
        "variance": float(np.var(samples)),
        "energy": float(np.sum(np.square(samples))),
        "entropy": float(-np.sum(probabilities * np.log(probabilities + _EPS))),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(samples)).astype(np.float64))),
        "waveform_length": float(np.sum(np.abs(np.diff(samples)))),
        "median_abs": float(np.median(abs_samples)),
    }


def _frequency_features(window: np.ndarray, config: PipelineConfig) -> dict[str, float]:
    freqs, amps = amplitude_spectrum(window, config.processed_fs)
    power = np.square(amps)
    power_sum = float(np.sum(power)) + _EPS
    weights = power / power_sum
    centroid = float(np.sum(freqs * weights))
    variance = float(np.sum(np.square(freqs - centroid) * weights))
    cumulative = np.cumsum(weights)
    band_map = {
        "10_500": _band_energy(freqs, power, 10.0, 500.0),
        "500_1000": _band_energy(freqs, power, 500.0, 1000.0),
        "1000_2000": _band_energy(freqs, power, 1000.0, 2000.0),
        "2000_5000": _band_energy(freqs, power, 2000.0, 5000.0),
    }
    positive = amps[1:] + _EPS
    return {
        "freq_centroid": centroid,
        "freq_rms": float(np.sqrt(np.sum(np.square(freqs) * weights))),
        "freq_std": float(np.sqrt(max(variance, 0.0))),
        "freq_skewness": float(np.sum(np.power(freqs - centroid, 3) * weights) / (np.power(np.sqrt(max(variance, 0.0)) + _EPS, 3))),
        "freq_kurtosis": float(np.sum(np.power(freqs - centroid, 4) * weights) / (np.power(max(variance, 0.0) + _EPS, 2))),
        "spectral_entropy": float(-np.sum(weights * np.log(weights + _EPS))),
        "peak_frequency": float(freqs[int(np.argmax(amps))]),
        "peak_amplitude": float(np.max(amps)),
        "band_energy_10_500": band_map["10_500"],
        "band_energy_500_1000": band_map["500_1000"],
        "band_energy_1000_2000": band_map["1000_2000"],
        "band_energy_2000_5000": band_map["2000_5000"],
        "band_ratio_10_500": band_map["10_500"] / power_sum,
        "band_ratio_500_1000": band_map["500_1000"] / power_sum,
        "band_ratio_1000_2000": band_map["1000_2000"] / power_sum,
        "band_ratio_2000_5000": band_map["2000_5000"] / power_sum,
        "spec_flatness": float(np.exp(np.mean(np.log(positive))) / (np.mean(positive) + _EPS)),
        "spec_rolloff_85": float(freqs[int(np.searchsorted(cumulative, 0.85, side="left"))]),
        "spec_rolloff_95": float(freqs[int(np.searchsorted(cumulative, 0.95, side="left"))]),
        "spec_flux": float(np.sum(np.square(np.diff(weights)))),
    }


def _rotation_features(window: np.ndarray, rpm: float | None, config: PipelineConfig) -> dict[str, float]:
    if rpm is None:
        return {name: 0.0 for name in ROTATION_FEATURES}
    freqs, amps = amplitude_spectrum(window, config.processed_fs)
    power = np.square(amps)
    base_freq = rpm / 60.0
    amp_1x = _sample_at(freqs, amps, base_freq)
    amp_2x = _sample_at(freqs, amps, base_freq * 2.0)
    amp_3x = _sample_at(freqs, amps, base_freq * 3.0)
    energy_1x = _band_energy(freqs, power, base_freq - config.harmonic_tolerance, base_freq + config.harmonic_tolerance)
    energy_2x = _band_energy(freqs, power, base_freq * 2.0 - config.harmonic_tolerance, base_freq * 2.0 + config.harmonic_tolerance)
    energy_3x = _band_energy(freqs, power, base_freq * 3.0 - config.harmonic_tolerance, base_freq * 3.0 + config.harmonic_tolerance)
    return {
        "amp_1x": amp_1x,
        "amp_2x": amp_2x,
        "amp_3x": amp_3x,
        "energy_1x": energy_1x,
        "energy_2x": energy_2x,
        "energy_3x": energy_3x,
        "ratio_2x_to_1x": amp_2x / (amp_1x + _EPS),
        "ratio_3x_to_1x": amp_3x / (amp_1x + _EPS),
        "rotation_band_energy": energy_1x + energy_2x + energy_3x,
    }


def _envelope_features(window: np.ndarray, rpm: float | None, config: PipelineConfig) -> dict[str, float]:
    samples = np.asarray(window, dtype=np.float64)
    envelope = np.abs(signal.hilbert(samples))
    envelope = envelope - np.mean(envelope)
    freqs, amps = amplitude_spectrum(envelope, config.processed_fs)
    amps_no_dc = amps.copy()
    if amps_no_dc.size:
        amps_no_dc[0] = 0.0
    power = np.square(amps_no_dc)
    power_sum = float(np.sum(power)) + _EPS
    weights = power / power_sum
    ratios = {
        "0_100": _band_energy(freqs, power, 0.0, 100.0) / power_sum,
        "100_500": _band_energy(freqs, power, 100.0, 500.0) / power_sum,
        "500_1000": _band_energy(freqs, power, 500.0, 1000.0) / power_sum,
        "1000_3000": _band_energy(freqs, power, 1000.0, 3000.0) / power_sum,
    }
    band_energies = {
        "0_100": _band_energy(freqs, power, 0.0, 100.0),
        "100_500": _band_energy(freqs, power, 100.0, 500.0),
        "500_1000": _band_energy(freqs, power, 500.0, 1000.0),
        "1000_3000": _band_energy(freqs, power, 1000.0, 3000.0),
    }
    peak_index = int(np.argmax(amps_no_dc))
    base_freq = None if rpm is None else rpm / 60.0
    env_amp_1x = 0.0 if base_freq is None else _sample_at(freqs, amps_no_dc, base_freq)
    env_amp_2x = 0.0 if base_freq is None else _sample_at(freqs, amps_no_dc, base_freq * 2.0)
    return {
        "env_rms": float(np.sqrt(np.mean(np.square(envelope)))),
        "env_peak": float(np.max(np.abs(envelope))),
        "env_kurtosis": float(stats.kurtosis(envelope, fisher=False, bias=False)),
        "env_entropy": float(_shannon_energy_entropy(envelope)),
        "env_peak_frequency": float(freqs[peak_index]),
        "env_peak_amplitude": float(amps_no_dc[peak_index]),
        "env_band_energy_0_100": band_energies["0_100"],
        "env_band_energy_100_500": band_energies["100_500"],
        "env_band_energy_500_1000": band_energies["500_1000"],
        "env_band_energy_1000_3000": band_energies["1000_3000"],
        "env_ratio_0_100": ratios["0_100"],
        "env_ratio_100_500": ratios["100_500"],
        "env_ratio_500_1000": ratios["500_1000"],
        "env_ratio_1000_3000": ratios["1000_3000"],
        "env_amp_1x": env_amp_1x,
        "env_amp_2x": env_amp_2x,
        "env_ratio_2x_to_1x": env_amp_2x / (env_amp_1x + _EPS),
        "env_spectral_entropy": float(-np.sum(weights * np.log(weights + _EPS))),
        "env_spec_centroid": float(np.sum(freqs * weights)),
        "env_spec_rms": float(np.sqrt(np.sum(np.square(freqs) * weights))),
    }


def _band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask]))


def _sample_at(freqs: np.ndarray, values: np.ndarray, target: float) -> float:
    return float(values[int(np.argmin(np.abs(freqs - target)))])


def _shannon_energy_entropy(samples: np.ndarray) -> float:
    energy = np.square(np.asarray(samples, dtype=np.float64))
    normalized = energy / (float(np.sum(energy)) + _EPS)
    return float(-np.sum(normalized * np.log(normalized + _EPS)))
