from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import signal

from pump_diagnosis.three_class_feature_table import FEATURE_COLUMNS


MOTOR2_FEATURE_TABLE = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_100_1480rpm_三分类_通道4_特征表.csv"
)
MOTOR4_FEATURE_TABLE = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor4_70_2070rpm_四分类_通道4_特征表.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/fault_feature_visualization"
)

_EPS = 1e-12


@dataclass(frozen=True)
class DeviceSpec:
    device: str
    feature_table: Path
    labels: tuple[str, ...]
    rpm: float

    @property
    def one_x_hz(self) -> float:
        return self.rpm / 60.0


@dataclass(frozen=True)
class VisualizationConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    original_fs: int = 20_000
    processed_fs: int = 12_000
    resample_up: int = 3
    resample_down: int = 5
    bandpass_low: float = 5.0
    bandpass_high: float = 5000.0
    filter_order: int = 4
    waveform_seconds: float = 1.0
    fft_low_max_hz: float = 300.0
    fft_full_max_hz: float = 5000.0
    harmonic_search_hz: float = 2.0
    envelope_band_low: float = 1000.0
    envelope_band_high: float = 5000.0
    envelope_max_hz: float = 1000.0
    wavelet: str = "db6"
    wavelet_level: int = 3
    plot_dpi: int = 300
    font_size: int = 11
    csv_encoding: str = "utf-8-sig"
    devices: tuple[DeviceSpec, ...] = field(
        default_factory=lambda: (
            DeviceSpec(
                device="Motor-2",
                feature_table=MOTOR2_FEATURE_TABLE,
                labels=("正常", "松动", "轴承故障"),
                rpm=1480.0,
            ),
            DeviceSpec(
                device="Motor-4",
                feature_table=MOTOR4_FEATURE_TABLE,
                labels=("正常", "转子不平衡", "联轴器不对中", "汽蚀"),
                rpm=2070.0,
            ),
        )
    )


@dataclass(frozen=True)
class FigureManifestEntry:
    figure_id: str
    device: str
    fault_label: str
    figure_type: str
    source_file: str
    record_index: str | int
    time_segment: str
    processing_params: str
    png_path: Path
    pdf_path: Path
    csv_path: Path


def build_figure_manifest_frame(entries: list[FigureManifestEntry], config: VisualizationConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_id": entry.figure_id,
                "device": entry.device,
                "fault_label": entry.fault_label,
                "figure_type": entry.figure_type,
                "source_file": entry.source_file,
                "record_index": entry.record_index,
                "time_segment": entry.time_segment,
                "processing_params": entry.processing_params,
                "png_path": str(entry.png_path),
                "pdf_path": str(entry.pdf_path),
                "csv_path": str(entry.csv_path),
            }
            for entry in entries
        ]
    )


def add_record_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "record_index" in enriched.columns and "group_id" in enriched.columns:
        return enriched
    record_index = enriched["window_id"].astype(str).str.rsplit("_", n=2).str[1].astype(int)
    enriched["record_index"] = record_index
    enriched["group_id"] = enriched["source_file"].astype(str) + "::record_" + enriched["record_index"].astype(str)
    return enriched


def select_representative_records(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, object]]:
    enriched = add_record_group_columns(frame)
    grouped = (
        enriched.groupby(["label", "group_id"], as_index=False)
        .agg(
            source_file=("source_file", "first"),
            record_index=("record_index", "first"),
            **{feature: (feature, "mean") for feature in feature_columns},
        )
    )
    selected: dict[str, dict[str, object]] = {}
    for label, label_frame in grouped.groupby("label", sort=False):
        feature_matrix = label_frame[feature_columns].to_numpy(dtype=float)
        center = feature_matrix.mean(axis=0)
        distances = np.linalg.norm(feature_matrix - center, axis=1)
        best_index = int(np.argmin(distances))
        row = label_frame.iloc[best_index]
        selected[label] = {
            "label": label,
            "group_id": row["group_id"],
            "source_file": row["source_file"],
            "record_index": int(row["record_index"]),
            "distance_to_center": float(distances[best_index]),
            "feature_center_json": json.dumps({feature: float(value) for feature, value in zip(feature_columns, center)}, ensure_ascii=False),
        }
    return selected


def run_visualization(config: VisualizationConfig) -> dict[str, object]:
    _configure_matplotlib(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[FigureManifestEntry] = []
    representative_rows: list[dict[str, object]] = []

    for spec in config.devices:
        feature_frame = add_record_group_columns(pd.read_csv(spec.feature_table))
        selected = select_representative_records(feature_frame, FEATURE_COLUMNS)
        device_signals: dict[str, dict[str, object]] = {}
        for label in spec.labels:
            sample = selected[label]
            representative_rows.append(
                {
                    "device": spec.device,
                    "fault_label": label,
                    "source_file": sample["source_file"],
                    "record_index": sample["record_index"],
                    "group_id": sample["group_id"],
                    "distance_to_center": sample["distance_to_center"],
                }
            )
            device_signals[label] = _build_signal_bundle(spec, sample, config)

        axes = _compute_device_axes(device_signals, spec, config)
        manifest_entries.extend(_write_device_figures(spec, device_signals, axes, config))

    manifest = build_figure_manifest_frame(manifest_entries, config)
    manifest.to_csv(config.output_root / "figure_manifest.csv", index=False, encoding=config.csv_encoding)
    pd.DataFrame(representative_rows).to_csv(
        config.output_root / "representative_records.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return {
        "figure_manifest": str(config.output_root / "figure_manifest.csv"),
        "representative_records": str(config.output_root / "representative_records.csv"),
        "figure_count": int(len(manifest_entries)),
    }


def _build_signal_bundle(spec: DeviceSpec, sample: dict[str, object], config: VisualizationConfig) -> dict[str, object]:
    raw = pd.read_csv(sample["source_file"])
    record_column = str(sample["record_index"])
    raw_signal = raw[record_column].to_numpy(dtype=float)
    processed = _preprocess_signal(raw_signal, config)
    waveform_samples = int(config.waveform_seconds * config.processed_fs)
    waveform = processed[:waveform_samples]
    waveform_time = np.arange(waveform.shape[0], dtype=float) / config.processed_fs
    fft_freqs, fft_amps = _hann_amplitude_spectrum(processed, config.processed_fs)
    envelope_signal = _envelope_signal(processed, config)
    env_freqs, env_amps = _hann_amplitude_spectrum(envelope_signal, config.processed_fs)
    wavelet_ratios = _wavelet_packet_ratios(processed, config)
    return {
        "sample": sample,
        "waveform_time": waveform_time,
        "waveform": waveform,
        "fft_freqs": fft_freqs,
        "fft_amps": fft_amps,
        "env_freqs": env_freqs,
        "env_amps": env_amps,
        "wavelet_ratios": wavelet_ratios,
        "one_x_hz": spec.one_x_hz,
    }


def _write_device_figures(
    spec: DeviceSpec,
    device_signals: dict[str, dict[str, object]],
    axes: dict[str, tuple[float, float]],
    config: VisualizationConfig,
) -> list[FigureManifestEntry]:
    entries: list[FigureManifestEntry] = []
    device_dir = config.output_root / spec.device
    figure_dirs = {
        "time_waveform": device_dir / "time_waveform",
        "fft_low": device_dir / "fft_low",
        "fft_full": device_dir / "fft_full",
        "envelope_spectrum": device_dir / "envelope_spectrum",
        "wavelet_bar": device_dir / "wavelet_bar",
        "wavelet_heatmap": device_dir / "wavelet_heatmap",
        "csv": device_dir / "csv",
    }
    for path in figure_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    heatmap_rows: list[dict[str, object]] = []
    for label in spec.labels:
        bundle = device_signals[label]
        sample = bundle["sample"]
        label_slug = _slugify_label(label)
        record_index = sample["record_index"]
        base_prefix = f"{spec.device.lower()}_{label_slug}"
        processing_params = _processing_params_text(spec, config)

        wave_csv = figure_dirs["csv"] / f"{base_prefix}_time_waveform.csv"
        pd.DataFrame({"time_s": bundle["waveform_time"], "amplitude": bundle["waveform"]}).to_csv(
            wave_csv, index=False, encoding=config.csv_encoding
        )
        wave_png = figure_dirs["time_waveform"] / f"{base_prefix}_time_waveform.png"
        wave_pdf = figure_dirs["time_waveform"] / f"{base_prefix}_time_waveform.pdf"
        _plot_waveform(bundle, label, spec, axes["time_waveform"], wave_png, wave_pdf, config)
        entries.append(
            FigureManifestEntry(
                figure_id=f"{base_prefix}_time_waveform",
                device=spec.device,
                fault_label=label,
                figure_type="time_waveform",
                source_file=sample["source_file"],
                record_index=record_index,
                time_segment=f"0.0-{config.waveform_seconds:.1f} s",
                processing_params=processing_params,
                png_path=wave_png,
                pdf_path=wave_pdf,
                csv_path=wave_csv,
            )
        )

        for figure_type, freq_max, axis_key in (
            ("fft_low", config.fft_low_max_hz, "fft_low"),
            ("fft_full", config.fft_full_max_hz, "fft_full"),
        ):
            mask = bundle["fft_freqs"] <= freq_max
            csv_path = figure_dirs["csv"] / f"{base_prefix}_{figure_type}.csv"
            pd.DataFrame(
                {
                    "frequency_hz": bundle["fft_freqs"][mask],
                    "amplitude": bundle["fft_amps"][mask],
                }
            ).to_csv(csv_path, index=False, encoding=config.csv_encoding)
            png_path = figure_dirs[figure_type] / f"{base_prefix}_{figure_type}.png"
            pdf_path = figure_dirs[figure_type] / f"{base_prefix}_{figure_type}.pdf"
            _plot_fft(bundle, label, spec, freq_max, axes[axis_key], png_path, pdf_path, config)
            entries.append(
                FigureManifestEntry(
                    figure_id=f"{base_prefix}_{figure_type}",
                    device=spec.device,
                    fault_label=label,
                    figure_type=figure_type,
                    source_file=sample["source_file"],
                    record_index=record_index,
                    time_segment="0.0-12.0 s",
                    processing_params=processing_params,
                    png_path=png_path,
                    pdf_path=pdf_path,
                    csv_path=csv_path,
                )
            )

        env_mask = bundle["env_freqs"] <= config.envelope_max_hz
        env_csv = figure_dirs["csv"] / f"{base_prefix}_envelope_spectrum.csv"
        pd.DataFrame(
            {
                "frequency_hz": bundle["env_freqs"][env_mask],
                "amplitude": bundle["env_amps"][env_mask],
            }
        ).to_csv(env_csv, index=False, encoding=config.csv_encoding)
        env_png = figure_dirs["envelope_spectrum"] / f"{base_prefix}_envelope_spectrum.png"
        env_pdf = figure_dirs["envelope_spectrum"] / f"{base_prefix}_envelope_spectrum.pdf"
        _plot_envelope(bundle, label, spec, axes["envelope_spectrum"], env_png, env_pdf, config)
        entries.append(
            FigureManifestEntry(
                figure_id=f"{base_prefix}_envelope_spectrum",
                device=spec.device,
                fault_label=label,
                figure_type="envelope_spectrum",
                source_file=sample["source_file"],
                record_index=record_index,
                time_segment="0.0-12.0 s",
                processing_params=processing_params,
                png_path=env_png,
                pdf_path=env_pdf,
                csv_path=env_csv,
            )
        )

        wavelet_csv = figure_dirs["csv"] / f"{base_prefix}_wavelet_bar.csv"
        wavelet_frame = pd.DataFrame(
            {"node": [f"节点{i}" for i in range(8)], "energy_ratio": bundle["wavelet_ratios"]}
        )
        wavelet_frame.to_csv(wavelet_csv, index=False, encoding=config.csv_encoding)
        wavelet_png = figure_dirs["wavelet_bar"] / f"{base_prefix}_wavelet_bar.png"
        wavelet_pdf = figure_dirs["wavelet_bar"] / f"{base_prefix}_wavelet_bar.pdf"
        _plot_wavelet_bar(bundle, label, spec, wavelet_png, wavelet_pdf, config)
        entries.append(
            FigureManifestEntry(
                figure_id=f"{base_prefix}_wavelet_bar",
                device=spec.device,
                fault_label=label,
                figure_type="wavelet_bar",
                source_file=sample["source_file"],
                record_index=record_index,
                time_segment="0.0-12.0 s",
                processing_params=processing_params,
                png_path=wavelet_png,
                pdf_path=wavelet_pdf,
                csv_path=wavelet_csv,
            )
        )

        heatmap_rows.append({"fault_label": label, **{f"节点{i}": float(v) for i, v in enumerate(bundle["wavelet_ratios"])}})

    heatmap_frame = pd.DataFrame(heatmap_rows).set_index("fault_label")
    heatmap_csv = figure_dirs["csv"] / f"{spec.device.lower()}_wavelet_heatmap.csv"
    heatmap_frame.to_csv(heatmap_csv, encoding=config.csv_encoding)
    heatmap_png = figure_dirs["wavelet_heatmap"] / f"{spec.device.lower()}_wavelet_heatmap.png"
    heatmap_pdf = figure_dirs["wavelet_heatmap"] / f"{spec.device.lower()}_wavelet_heatmap.pdf"
    _plot_wavelet_heatmap(heatmap_frame, spec, heatmap_png, heatmap_pdf, config)
    entries.append(
        FigureManifestEntry(
            figure_id=f"{spec.device.lower()}_wavelet_heatmap",
            device=spec.device,
            fault_label="ALL",
            figure_type="wavelet_heatmap",
            source_file="; ".join(str(device_signals[label]["sample"]["source_file"]) for label in spec.labels),
            record_index="; ".join(str(device_signals[label]["sample"]["record_index"]) for label in spec.labels),
            time_segment="0.0-12.0 s",
            processing_params=_processing_params_text(spec, config),
            png_path=heatmap_png,
            pdf_path=heatmap_pdf,
            csv_path=heatmap_csv,
        )
    )
    return entries


def _compute_device_axes(
    device_signals: dict[str, dict[str, object]],
    spec: DeviceSpec,
    config: VisualizationConfig,
) -> dict[str, tuple[float, float]]:
    waveform_values = np.concatenate([bundle["waveform"] for bundle in device_signals.values()])
    fft_low_max = max(
        float(np.max(bundle["fft_amps"][bundle["fft_freqs"] <= config.fft_low_max_hz]))
        for bundle in device_signals.values()
    )
    fft_full_max = max(
        float(np.max(bundle["fft_amps"][bundle["fft_freqs"] <= config.fft_full_max_hz]))
        for bundle in device_signals.values()
    )
    envelope_max = max(
        float(np.max(bundle["env_amps"][bundle["env_freqs"] <= config.envelope_max_hz]))
        for bundle in device_signals.values()
    )
    return {
        "time_waveform": (float(np.min(waveform_values)), float(np.max(waveform_values))),
        "fft_low": (0.0, fft_low_max * 1.05 + _EPS),
        "fft_full": (0.0, fft_full_max * 1.05 + _EPS),
        "envelope_spectrum": (0.0, envelope_max * 1.05 + _EPS),
    }


def _preprocess_signal(samples: np.ndarray, config: VisualizationConfig) -> np.ndarray:
    centered = np.asarray(samples, dtype=float) - np.mean(samples)
    resampled = signal.resample_poly(centered, up=config.resample_up, down=config.resample_down)
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.bandpass_low, config.bandpass_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, resampled)
    return np.asarray(filtered - np.mean(filtered), dtype=float)


def _hann_amplitude_spectrum(samples: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    signal_array = np.asarray(samples, dtype=float)
    window = np.hanning(signal_array.shape[0])
    weighted = signal_array * window
    spectrum = np.fft.rfft(weighted)
    amplitudes = np.abs(spectrum) * 2.0 / (np.sum(window) + _EPS)
    frequencies = np.fft.rfftfreq(signal_array.shape[0], d=1.0 / fs)
    return frequencies, amplitudes


def _envelope_signal(samples: np.ndarray, config: VisualizationConfig) -> np.ndarray:
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.envelope_band_low, config.envelope_band_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    band_limited = signal.sosfiltfilt(sos, samples)
    envelope = np.abs(signal.hilbert(band_limited))
    return np.asarray(envelope - np.mean(envelope), dtype=float)


def _wavelet_packet_ratios(samples: np.ndarray, config: VisualizationConfig) -> np.ndarray:
    wp = pywt.WaveletPacket(data=np.asarray(samples, dtype=float), wavelet=config.wavelet, mode="symmetric", maxlevel=config.wavelet_level)
    nodes = wp.get_level(config.wavelet_level, order="freq")
    energies = np.asarray([float(np.sum(np.square(node.data))) for node in nodes], dtype=float)
    return energies / (float(np.sum(energies)) + _EPS)


def _plot_waveform(bundle: dict[str, object], label: str, spec: DeviceSpec, y_limits: tuple[float, float], png_path: Path, pdf_path: Path, config: VisualizationConfig) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(bundle["waveform_time"], bundle["waveform"], color="#1f4e79", linewidth=1.0)
    ax.set_xlim(0.0, config.waveform_seconds)
    ax.set_ylim(y_limits)
    ax.set_title(f"{spec.device} {label} 时域波形")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("幅值 / a.u.")
    ax.grid(True, alpha=0.25)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)


def _plot_fft(bundle: dict[str, object], label: str, spec: DeviceSpec, freq_max: float, y_limits: tuple[float, float], png_path: Path, pdf_path: Path, config: VisualizationConfig) -> None:
    mask = bundle["fft_freqs"] <= freq_max
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(bundle["fft_freqs"][mask], bundle["fft_amps"][mask], color="#c55a11", linewidth=1.0)
    _annotate_harmonics(ax, spec.one_x_hz, freq_max, y_limits[1], color="#7f6000")
    ax.set_xlim(0.0, freq_max)
    ax.set_ylim(y_limits)
    ax.set_title(f"{spec.device} {label} FFT频谱 ({int(freq_max)} Hz)")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("幅值 / a.u.")
    ax.grid(True, alpha=0.25)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)


def _plot_envelope(bundle: dict[str, object], label: str, spec: DeviceSpec, y_limits: tuple[float, float], png_path: Path, pdf_path: Path, config: VisualizationConfig) -> None:
    mask = bundle["env_freqs"] <= config.envelope_max_hz
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(bundle["env_freqs"][mask], bundle["env_amps"][mask], color="#2f7d32", linewidth=1.0)
    _annotate_harmonics(ax, spec.one_x_hz, config.envelope_max_hz, y_limits[1], color="#1c5d1f")
    ax.set_xlim(0.0, config.envelope_max_hz)
    ax.set_ylim(y_limits)
    ax.set_title(f"{spec.device} {label} 包络谱")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("幅值 / a.u.")
    ax.grid(True, alpha=0.25)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)


def _plot_wavelet_bar(bundle: dict[str, object], label: str, spec: DeviceSpec, png_path: Path, pdf_path: Path, config: VisualizationConfig) -> None:
    nodes = [f"节点{i}" for i in range(8)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(nodes, bundle["wavelet_ratios"], color="#5b9bd5", edgecolor="#2f5597")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{spec.device} {label} 小波包能量分布")
    ax.set_xlabel("频带节点")
    ax.set_ylabel("能量占比")
    ax.grid(True, axis="y", alpha=0.25)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)


def _plot_wavelet_heatmap(frame: pd.DataFrame, spec: DeviceSpec, png_path: Path, pdf_path: Path, config: VisualizationConfig) -> None:
    matrix = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(frame.shape[1]))
    ax.set_xticklabels(frame.columns)
    ax.set_yticks(np.arange(frame.shape[0]))
    ax.set_yticklabels(frame.index)
    ax.set_title(f"{spec.device} 各故障类别小波包能量热力图")
    ax.set_xlabel("频带节点")
    ax.set_ylabel("故障类别")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="black", fontsize=config.font_size - 1)
    fig.colorbar(image, ax=ax, label="能量占比")
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)


def _annotate_harmonics(ax: plt.Axes, one_x_hz: float, freq_max: float, y_max: float, color: str) -> None:
    for multiple in (1, 2, 3):
        harmonic = one_x_hz * multiple
        if harmonic > freq_max:
            continue
        ax.axvline(harmonic, color=color, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(harmonic, y_max * 0.96, f"{multiple}X", rotation=90, va="top", ha="center", color=color)


def _configure_matplotlib(config: VisualizationConfig) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = config.font_size
    plt.rcParams["axes.titlesize"] = config.font_size + 1
    plt.rcParams["axes.labelsize"] = config.font_size
    plt.rcParams["xtick.labelsize"] = config.font_size - 1
    plt.rcParams["ytick.labelsize"] = config.font_size - 1
    plt.rcParams["legend.fontsize"] = config.font_size - 1


def _save_dual(fig: plt.Figure, png_path: Path, pdf_path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)


def _processing_params_text(spec: DeviceSpec, config: VisualizationConfig) -> str:
    return (
        f"device={spec.device}; fs={config.original_fs}->{config.processed_fs} Hz; "
        f"bandpass={config.bandpass_low}-{config.bandpass_high} Hz; "
        f"envelope_band={config.envelope_band_low}-{config.envelope_band_high} Hz; "
        f"wavelet={config.wavelet}; level={config.wavelet_level}; "
        f"1X={spec.one_x_hz:.2f} Hz"
    )


def _slugify_label(label: str) -> str:
    mapping = {
        "正常": "normal",
        "松动": "looseness",
        "轴承故障": "bearing_fault",
        "转子不平衡": "rotor_unbalance",
        "联轴器不对中": "coupling_misalignment",
        "汽蚀": "cavitation",
    }
    return mapping.get(label, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready fault feature visualization figures.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    config = VisualizationConfig(output_root=Path(args.output_root))
    result = run_visualization(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
