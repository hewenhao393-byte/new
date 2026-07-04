from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats


DEFAULT_DATA_ROOT = Path(
    "/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/100"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_100_1480rpm_三分类_通道4_特征表.csv"
)

FEATURE_COLUMNS = [
    "rms",
    "variance",
    "standard_deviation",
    "peak_to_peak",
    "absolute_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_1x_amp",
    "rot_2x_amp",
    "rot_3x_amp",
    "rot_1x_energy",
    "rot_2x_energy",
    "rot_3x_energy",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "dominant_frequency_ratio",
    "energy_ratio_0_1000",
    "energy_ratio_1000_3000",
    "energy_ratio_3000_5000",
    "high_low_energy_ratio",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "env_rms",
    "env_kurtosis",
    "env_crest_factor",
    "env_spectral_peak",
    "env_spectral_entropy",
    "env_energy_ratio_1000_3000",
    "env_energy_ratio_3000_5000",
]
OUTPUT_COLUMNS = ["source_file", "window_id", "window_start", "window_end", "label"] + FEATURE_COLUMNS

_EPS = 1e-12

_LABEL_PREFIXES = {
    "正常": ("正常状态",),
    "松动": ("软脚", "电机地脚松动", "泵地脚松动"),
    "轴承故障": ("轴承内圈故障", "轴承外圈故障", "轴承滚动体故障", "轴承污染", "泵轴承故障"),
}


@dataclass(frozen=True)
class ThreeClassFeatureConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_path: Path = DEFAULT_OUTPUT_PATH
    original_fs: int = 20_000
    processed_fs: int = 12_000
    resample_up: int = 3
    resample_down: int = 5
    bandpass_low: float = 10.0
    bandpass_high: float = 5000.0
    filter_order: int = 4
    window_size: int = 2400
    step_size: int = 1200
    rpm: float = 1480.0
    harmonic_search_hz: float = 2.0
    wavelet: str = "db6"
    wavelet_level: int = 3
    envelope_band_low: float = 2000.0
    envelope_band_high: float = 5000.0
    csv_encoding: str = "utf-8-sig"
    summary_path: Path | None = field(default=None)

    @property
    def rotation_frequency(self) -> float:
        return self.rpm / 60.0


def map_fault_to_three_class(raw_fault_name: str) -> str | None:
    for label, prefixes in _LABEL_PREFIXES.items():
        if any(raw_fault_name.startswith(prefix) for prefix in prefixes):
            return label
    return None


def build_feature_table(config: ThreeClassFeatureConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(config.data_root.rglob("*通道4.csv")):
        label = map_fault_to_three_class(csv_path.parent.name)
        if label is None:
            continue
        raw = pd.read_csv(csv_path)
        signal_columns = [column for column in raw.columns if column != "time"]
        for column_name in signal_columns:
            samples = raw[column_name].to_numpy(dtype=np.float64)
            processed = preprocess_signal(samples, config)
            for window_index, (start, end, window) in enumerate(iter_windows(processed, config.window_size, config.step_size)):
                row = {
                    "source_file": str(csv_path),
                    "window_id": f"{csv_path.stem}_{column_name}_{window_index}",
                    "window_start": int(start),
                    "window_end": int(end),
                    "label": label,
                }
                row.update(extract_window_features(window, config))
                rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_feature_table(config: ThreeClassFeatureConfig) -> tuple[Path, pd.DataFrame]:
    frame = build_feature_table(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output_path, index=False, encoding=config.csv_encoding)
    summary_path = config.summary_path or config.output_path.with_suffix(".summary.json")
    summary = {
        "rows": int(len(frame)),
        "files": int(frame["source_file"].nunique()) if not frame.empty else 0,
        "labels": frame["label"].value_counts().to_dict() if not frame.empty else {},
        "window_size": config.window_size,
        "step_size": config.step_size,
        "processed_fs": config.processed_fs,
        "rpm": config.rpm,
        "wavelet": config.wavelet,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return config.output_path, frame


def preprocess_signal(samples: np.ndarray, config: ThreeClassFeatureConfig) -> np.ndarray:
    centered = np.asarray(samples, dtype=np.float64) - np.mean(samples)
    resampled = signal.resample_poly(centered, up=config.resample_up, down=config.resample_down)
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.bandpass_low, config.bandpass_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, resampled)
    return np.asarray(filtered, dtype=np.float64) - np.mean(filtered)


def iter_windows(samples: np.ndarray, window_size: int, step_size: int):
    for start in range(0, samples.shape[0] - window_size + 1, step_size):
        end = start + window_size
        yield start, end, samples[start:end]


def extract_window_features(window: np.ndarray, config: ThreeClassFeatureConfig) -> dict[str, float]:
    samples = np.asarray(window, dtype=np.float64)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    absolute_peak = float(np.max(abs_samples))
    mean_abs = float(np.mean(abs_samples))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_samples + _EPS)))

    freqs, amps = amplitude_spectrum(samples, config.processed_fs)
    power = np.square(amps)
    total_power = float(np.sum(power)) + _EPS
    spectral_weights = power / total_power

    fr = config.rotation_frequency
    harmonic_amps = []
    harmonic_energies = []
    for multiple in (1, 2, 3):
        center = fr * multiple
        harmonic_amps.append(_band_max_amplitude(freqs, amps, center - config.harmonic_search_hz, center + config.harmonic_search_hz))
        harmonic_energies.append(_band_energy(freqs, power, center - config.harmonic_search_hz, center + config.harmonic_search_hz))

    harmonic_energy_1x_5x = sum(
        _band_energy(freqs, power, fr * multiple - config.harmonic_search_hz, fr * multiple + config.harmonic_search_hz)
        for multiple in range(1, 6)
    )

    wp = pywt.WaveletPacket(data=samples, wavelet=config.wavelet, mode="symmetric", maxlevel=config.wavelet_level)
    nodes = wp.get_level(config.wavelet_level, order="freq")
    node_energies = np.asarray([float(np.sum(np.square(node.data))) for node in nodes], dtype=np.float64)
    node_ratios = node_energies / (float(np.sum(node_energies)) + _EPS)

    envelope = _envelope_signal(samples, config)
    env_abs = np.abs(envelope)
    env_rms = float(np.sqrt(np.mean(np.square(envelope))))
    env_peak = float(np.max(env_abs))
    env_freqs, env_amps = amplitude_spectrum(envelope, config.processed_fs)
    if env_amps.size:
        env_amps = env_amps.copy()
        env_amps[0] = 0.0
    env_power = np.square(env_amps)
    env_total_power = float(np.sum(env_power)) + _EPS
    env_weights = env_power / env_total_power

    feature_map = {
        "rms": rms,
        "variance": float(np.var(samples)),
        "standard_deviation": float(np.std(samples)),
        "peak_to_peak": float(np.ptp(samples)),
        "absolute_peak": absolute_peak,
        "kurtosis": float(stats.kurtosis(samples, fisher=False, bias=False)),
        "skewness": float(stats.skew(samples, bias=False)),
        "crest_factor": absolute_peak / (rms + _EPS),
        "impulse_factor": absolute_peak / (mean_abs + _EPS),
        "clearance_factor": absolute_peak / (sqrt_abs_mean**2 + _EPS),
        "shape_factor": rms / (mean_abs + _EPS),
        "rot_1x_amp": harmonic_amps[0],
        "rot_2x_amp": harmonic_amps[1],
        "rot_3x_amp": harmonic_amps[2],
        "rot_1x_energy": harmonic_energies[0],
        "rot_2x_energy": harmonic_energies[1],
        "rot_3x_energy": harmonic_energies[2],
        "rot_2x_1x_ratio": harmonic_amps[1] / (harmonic_amps[0] + _EPS),
        "rot_3x_1x_ratio": harmonic_amps[2] / (harmonic_amps[0] + _EPS),
        "harmonic_energy_ratio_1x_5x": harmonic_energy_1x_5x / total_power,
        "spectral_centroid": float(np.sum(freqs * spectral_weights)),
        "spectral_entropy": float(-np.sum(spectral_weights * np.log(spectral_weights + _EPS))),
        "spectral_flatness": _spectral_flatness(power[1:]),
        "dominant_frequency_ratio": float(np.max(power)) / total_power,
        "energy_ratio_0_1000": _band_energy(freqs, power, 0.0, 1000.0) / total_power,
        "energy_ratio_1000_3000": _band_energy(freqs, power, 1000.0, 3000.0) / total_power,
        "energy_ratio_3000_5000": _band_energy(freqs, power, 3000.0, 5000.0) / total_power,
        "high_low_energy_ratio": _band_energy(freqs, power, 3000.0, 5000.0) / (_band_energy(freqs, power, 0.0, 1000.0) + _EPS),
        "env_rms": env_rms,
        "env_kurtosis": float(stats.kurtosis(envelope, fisher=False, bias=False)),
        "env_crest_factor": env_peak / (env_rms + _EPS),
        "env_spectral_peak": float(env_freqs[int(np.argmax(env_amps))]) if env_amps.size else 0.0,
        "env_spectral_entropy": float(-np.sum(env_weights * np.log(env_weights + _EPS))),
        "env_energy_ratio_1000_3000": _band_energy(env_freqs, env_power, 1000.0, 3000.0) / env_total_power,
        "env_energy_ratio_3000_5000": _band_energy(env_freqs, env_power, 3000.0, 5000.0) / env_total_power,
    }
    for index, ratio in enumerate(node_ratios):
        feature_map[f"wp_energy_ratio_{index}"] = float(ratio)
    return {name: float(feature_map[name]) for name in FEATURE_COLUMNS}


def amplitude_spectrum(window: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
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


def _envelope_signal(samples: np.ndarray, config: ThreeClassFeatureConfig) -> np.ndarray:
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.envelope_band_low, config.envelope_band_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    band_limited = signal.sosfiltfilt(sos, samples)
    envelope = np.abs(signal.hilbert(band_limited))
    return np.asarray(envelope - np.mean(envelope), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Motor-2 1480 rpm channel-4 three-class feature table.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--wavelet", choices=["db4", "db6"], default="db6")
    args = parser.parse_args()

    config = ThreeClassFeatureConfig(
        data_root=Path(args.data_root),
        output_path=Path(args.output_path),
        wavelet=args.wavelet,
    )
    output_path, frame = save_feature_table(config)
    print(f"saved={output_path}")
    print(f"rows={len(frame)}")
    if not frame.empty:
        print(frame['label'].value_counts().to_string())


if __name__ == "__main__":
    main()
