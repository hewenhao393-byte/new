from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pywt
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_V3_CONTRACT
from pump_fault_app.domain.records import FeatureVector, SampleMetadata


EPSILON = 1e-12
ORDER_HALF_WIDTH_HZ = 2.5


class OutOfBandOrderError(ValueError):
    pass


def _order_band(center_hz: float) -> tuple[float, float]:
    low_hz = FORMAL_V3_CONTRACT.filter_low_hz
    high_hz = FORMAL_V3_CONTRACT.filter_high_hz
    if not low_hz <= center_hz <= high_hz:
        raise OutOfBandOrderError(f"order center {center_hz} outside 5-5000 Hz")
    return max(center_hz - ORDER_HALF_WIDTH_HZ, low_hz), min(center_hz + ORDER_HALF_WIDTH_HZ, high_hz)


def _spectrum(samples: np.ndarray, low_hz: float = 5.0, high_hz: float = 5000.0):
    sample_rate = FORMAL_V3_CONTRACT.target_sampling_rate
    frequency = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    mask = (frequency >= low_hz) & (frequency <= high_hz)
    return frequency[mask], power[mask]


def _entropy(power: np.ndarray) -> float:
    normalized = power / (power.sum() + EPSILON)
    if len(normalized) <= 1:
        return 0.0
    return float(-np.sum(normalized * np.log(normalized + EPSILON)) / np.log(len(normalized)))


def _band_energy(
    frequency: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    include_high: bool = True,
) -> float:
    mask = (frequency >= low_hz) & ((frequency <= high_hz) if include_high else (frequency < high_hz))
    return float(power[mask].sum())


def _extract_accepted_feature_map(samples: np.ndarray, rpm: float) -> OrderedDict[str, float]:
    contract = FORMAL_V3_CONTRACT
    if samples.shape != (contract.window_size,) or not np.isfinite(samples).all():
        raise ValueError("window must contain 4800 finite samples")
    if not np.isfinite(rpm) or rpm <= 0.0:
        raise ValueError("rpm must be a positive finite value")

    output: OrderedDict[str, float] = OrderedDict()
    absolute = np.abs(samples)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(absolute.max())
    mean_absolute = float(absolute.mean())
    output.update(
        rms=rms,
        std=float(np.std(samples)),
        peak_to_peak=float(np.ptp(samples)),
        skewness=float(skew(samples, bias=False)),
        kurtosis=float(kurtosis(samples, fisher=False, bias=False)),
        crest_factor=peak / (rms + EPSILON),
        impulse_factor=peak / (mean_absolute + EPSILON),
        clearance_factor=peak / (float(np.mean(np.sqrt(absolute))) ** 2 + EPSILON),
        shape_factor=rms / (mean_absolute + EPSILON),
    )

    frequency, power = _spectrum(samples)
    total_energy = float(power.sum())
    rotation_hz = float(rpm) / 60.0
    orders = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5)
    order_energies = {
        order: _band_energy(frequency, power, *_order_band(order * rotation_hz)) for order in orders
    }
    harmonic_energy = sum(order_energies[order] for order in (1, 2, 3, 4, 5))
    output.update(
        rot_1x_energy_ratio=order_energies[1] / (total_energy + EPSILON),
        rot_2x_energy_ratio=order_energies[2] / (total_energy + EPSILON),
        rot_3x_energy_ratio=order_energies[3] / (total_energy + EPSILON),
        rot_2x_1x_ratio=order_energies[2] / (order_energies[1] + EPSILON),
        rot_3x_1x_ratio=order_energies[3] / (order_energies[1] + EPSILON),
        harmonic_energy_ratio_1x_5x=harmonic_energy / (total_energy + EPSILON),
        harmonic_energy_ratio_3x_5x=sum(order_energies[order] for order in (3, 4, 5))
        / (harmonic_energy + EPSILON),
        rot_2x_harmonic_ratio=order_energies[2] / (harmonic_energy + EPSILON),
        rot_05x_1x_ratio=order_energies[0.5] / (order_energies[1] + EPSILON),
        noninteger_harmonic_energy_ratio=sum(order_energies[order] for order in (0.5, 1.5, 2.5))
        / (total_energy + EPSILON),
    )

    centroid = float(np.sum(frequency * power) / (total_energy + EPSILON))
    output.update(
        spectral_entropy=_entropy(power),
        spectral_flatness=float(
            np.exp(np.mean(np.log(power + EPSILON))) / (np.mean(power) + EPSILON)
        ),
        spectral_centroid=centroid,
        spectral_bandwidth=float(
            np.sqrt(np.sum(((frequency - centroid) ** 2) * power) / (total_energy + EPSILON))
        ),
    )

    energy_5_300 = _band_energy(frequency, power, 5, 300, include_high=False)
    energy_300_1000 = _band_energy(frequency, power, 300, 1000, include_high=False)
    energy_1000_3000 = _band_energy(frequency, power, 1000, 3000, include_high=False)
    energy_3000_5000 = _band_energy(frequency, power, 3000, 5000)
    partition = energy_5_300 + energy_300_1000 + energy_1000_3000 + energy_3000_5000
    output.update(
        band_energy_5_300_ratio=energy_5_300 / (partition + EPSILON),
        band_energy_300_1000_ratio=energy_300_1000 / (partition + EPSILON),
        band_energy_1000_3000_ratio=energy_1000_3000 / (partition + EPSILON),
        band_energy_3000_5000_ratio=energy_3000_5000 / (partition + EPSILON),
        high_low_energy_ratio=(energy_1000_3000 + energy_3000_5000) / (energy_5_300 + EPSILON),
    )

    nodes = pywt.WaveletPacket(
        samples,
        contract.wavelet,
        maxlevel=contract.wavelet_level,
    ).get_level(contract.wavelet_level, order="natural")
    wavelet_energy = np.asarray([np.sum(np.square(node.data)) for node in nodes], dtype=np.float64)
    wavelet_ratio = wavelet_energy / (wavelet_energy.sum() + EPSILON)
    for index, value in enumerate(wavelet_ratio):
        output[f"wp_energy_ratio_{index}"] = float(value)
    output["wp_energy_entropy"] = float(
        -np.sum(wavelet_ratio * np.log(wavelet_ratio + EPSILON)) / np.log(8)
    )

    sos = butter(4, [1000, 5000], btype="bandpass", fs=contract.target_sampling_rate, output="sos")
    envelope = np.abs(hilbert(sosfiltfilt(sos, samples)))
    envelope = envelope - envelope.mean()
    envelope_frequency, envelope_power = _spectrum(envelope, 5, 500)
    envelope_total = float(envelope_power.sum())
    resolution_hz = contract.target_sampling_rate / contract.window_size
    peaks, _ = find_peaks(
        envelope_power,
        prominence=(float(envelope_power.max()) * 0.05 if len(envelope_power) else 0.0),
        distance=max(1, int(np.ceil(5 / resolution_hz))),
    )
    if len(peaks) > 5:
        peaks = peaks[np.argsort(envelope_power[peaks])[-5:]]
    peak_energies = [
        _band_energy(
            envelope_frequency,
            envelope_power,
            max(5, envelope_frequency[index] - 2.5),
            min(500, envelope_frequency[index] + 2.5),
        )
        for index in peaks
    ]
    peak_energy_sum = sum(peak_energies)
    output.update(
        env_kurtosis=float(kurtosis(envelope, fisher=False, bias=False)),
        env_crest_factor=float(
            np.max(np.abs(envelope)) / (np.sqrt(np.mean(envelope * envelope)) + EPSILON)
        ),
        env_spectral_entropy=_entropy(envelope_power),
        env_peak_energy_ratio=peak_energy_sum / (envelope_total + EPSILON),
        env_peak_concentration=(
            max(peak_energies) / (peak_energy_sum + EPSILON) if peak_energies else 0.0
        ),
        env_peak_count=float(len(peaks)),
    )
    return output


def extract_catboost43_features(samples: np.ndarray, rpm: float) -> FeatureVector:
    feature_map = _extract_accepted_feature_map(np.asarray(samples, dtype=np.float64), rpm)
    if tuple(feature_map) != FORMAL_FEATURE_NAMES:
        raise RuntimeError("feature order mismatch")
    values = tuple(float(feature_map[name]) for name in FORMAL_FEATURE_NAMES)
    if not np.isfinite(values).all():
        raise ValueError("non-finite feature")
    return FeatureVector(
        feature_names=FORMAL_FEATURE_NAMES,
        values=values,
        metadata=SampleMetadata(rpm=rpm),
    )
