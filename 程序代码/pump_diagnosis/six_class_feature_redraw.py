from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pump_diagnosis.fault_feature_visualization import (
    VisualizationConfig,
    _configure_matplotlib,
    _envelope_signal,
    _hann_amplitude_spectrum,
    _preprocess_signal,
    _save_dual,
    _wavelet_packet_ratios,
)
from pump_diagnosis.unified_six_class_exploration_v2 import MOTOR4_RPM, RPM_BY_SPEED, add_speed_metadata


RESULTS_ROOT = Path("/Users/hewenhao/Documents/特征提取/实验结果")
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "six_class_feature_redraw"
MOTOR2_FEATURE_TABLES = (
    RESULTS_ROOT / "Motor2_50_740rpm_三分类_通道4_特征表.csv",
    RESULTS_ROOT / "Motor2_75_1110rpm_三分类_通道4_特征表.csv",
    RESULTS_ROOT / "Motor2_100_1480rpm_三分类_通道4_特征表.csv",
)
MOTOR4_FEATURE_TABLE = RESULTS_ROOT / "Motor4_70_2070rpm_四分类_通道4_特征表.csv"
MANIFEST_REQUIRED_COLUMNS = [
    "label",
    "device_id",
    "speed_percent",
    "rpm",
    "source_file",
    "group_id",
    "window_id",
]
ENVELOPE_EXPAND_LIMIT_HZ = 300.0
FULL_SPECTRUM_DB_FLOOR = -80.0
_EPS = 1e-12


CLASS_LAYOUT_ORDER = [
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
]

REDRAW_FEATURE_COLUMNS = [
    "kurtosis",
    "skewness",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "spectral_entropy",
    "spectral_flatness",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "env_kurtosis",
    "env_crest_factor",
]


@dataclass(frozen=True)
class RedrawConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    original_fs: int = 20_000
    processed_fs: int = 12_000
    resample_up: int = 3
    resample_down: int = 5
    bandpass_low: float = 10.0
    bandpass_high: float = 5000.0
    filter_order: int = 4
    window_size: int = 2400
    window_step: int = 1200
    fft_low_max_hz: float = 300.0
    fft_full_max_hz: float = 5000.0
    envelope_default_max_hz: float = 300.0
    envelope_expanded_max_hz: float = 500.0
    envelope_band_low: float = 2000.0
    envelope_band_high: float = 5000.0
    wavelet: str = "db6"
    wavelet_level: int = 3
    plot_dpi: int = 300
    font_size: int = 11
    page_title_size: float = 11.5
    panel_title_size: float = 10.8
    axis_label_size: float = 9.5
    tick_label_size: float = 8.8
    csv_encoding: str = "utf-8-sig"


def select_representative_windows(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    missing_columns = [column for column in REDRAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"frame is missing required redraw feature columns: {missing_list}")

    selected: dict[str, dict[str, object]] = {}
    if frame.empty:
        return selected

    missing_labels = [label for label in CLASS_LAYOUT_ORDER if label not in set(frame["label"])]
    if missing_labels:
        missing_list = ", ".join(missing_labels)
        raise ValueError(f"frame is missing required redraw labels: {missing_list}")

    feature_frame = frame.loc[:, REDRAW_FEATURE_COLUMNS].astype(float)
    non_finite_mask = ~np.isfinite(feature_frame.to_numpy(dtype=float))
    if non_finite_mask.any():
        bad_rows = np.flatnonzero(non_finite_mask.any(axis=1))
        raise ValueError(f"frame contains non-finite redraw feature values at rows: {bad_rows.tolist()}")

    for label, label_frame in frame.groupby("label", sort=False):
        label_features = feature_frame.loc[label_frame.index]
        center = label_features.mean(axis=0).to_numpy(dtype=float)
        distances = np.linalg.norm(label_features.to_numpy(dtype=float) - center, axis=1)
        best_pos = int(np.argmin(distances))
        row = label_frame.iloc[best_pos].to_dict()
        row["distance_to_center"] = float(distances[best_pos])
        selected[label] = row

    ordered: dict[str, dict[str, object]] = {}
    for label in CLASS_LAYOUT_ORDER:
        if label in selected:
            ordered[label] = selected[label]
    for label, row in selected.items():
        if label not in ordered:
            ordered[label] = row
    return ordered


def normalize_window_waveform(waveform: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(waveform, dtype=float)
    if values.size == 0:
        return values
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return values.copy()
    return values / scale


def should_expand_envelope_limit(peaks_hz: Iterable[float]) -> bool:
    return any(float(peak) > ENVELOPE_EXPAND_LIMIT_HZ for peak in peaks_hz)


def write_representative_window_manifest(rows: Iterable[dict[str, object]], output: Path) -> Path:
    frame = pd.DataFrame(list(rows))
    missing_columns = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"representative manifest rows are missing required columns: {missing_list}")

    ordered_columns = MANIFEST_REQUIRED_COLUMNS + [column for column in frame.columns if column not in MANIFEST_REQUIRED_COLUMNS]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ordered_columns].to_csv(output, index=False, encoding="utf-8-sig")
    return output


def run_six_class_feature_redraw(config: RedrawConfig) -> dict[str, str]:
    _configure_matplotlib(_to_visualization_config(config))
    config.output_root.mkdir(parents=True, exist_ok=True)

    feature_frame = _load_unified_feature_frame()
    selected = select_representative_windows(feature_frame)
    bundles = {label: _build_window_bundle(selected[label], config) for label in CLASS_LAYOUT_ORDER}

    representative_rows = [_manifest_row(bundle["metadata"]) for bundle in bundles.values()]
    representative_path = write_representative_window_manifest(
        representative_rows,
        config.output_root / "representative_windows.csv",
    )
    parameter_path = _write_parameter_summary(config, bundles)

    time_csv = _write_time_waveform_figure(bundles, config)
    low_csv = _write_low_frequency_figure(bundles, config)
    full_csv = _write_full_spectrum_figure(bundles, config)
    wavelet_csv = _write_wavelet_figure(bundles, config)
    envelope_csv = _write_envelope_figure(bundles, config)

    return {
        "output_root": str(config.output_root),
        "representative_windows": str(representative_path),
        "processing_parameters": str(parameter_path),
        "fig4_2_time_waveform": str(config.output_root / "fig4_2_time_waveform.png"),
        "fig4_2_time_waveform_csv": str(time_csv),
        "fig4_3_low_frequency_spectrum": str(config.output_root / "fig4_3_low_frequency_spectrum.png"),
        "fig4_3_low_frequency_spectrum_csv": str(low_csv),
        "fig4_4_full_spectrum": str(config.output_root / "fig4_4_full_spectrum.png"),
        "fig4_4_full_spectrum_csv": str(full_csv),
        "fig4_5_wavelet_packet_bar": str(config.output_root / "fig4_5_wavelet_packet_bar.png"),
        "fig4_5_wavelet_packet_bar_csv": str(wavelet_csv),
        "fig4_6_envelope_spectrum": str(config.output_root / "fig4_6_envelope_spectrum.png"),
        "fig4_6_envelope_spectrum_csv": str(envelope_csv),
    }


def _load_unified_feature_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in MOTOR2_FEATURE_TABLES:
        frame = pd.read_csv(path).copy()
        frame["device_id"] = "Motor-2"
        frames.append(frame)
    motor4_frame = pd.read_csv(MOTOR4_FEATURE_TABLE).copy()
    motor4_frame["device_id"] = "Motor-4"
    frames.append(motor4_frame)

    merged = add_speed_metadata(pd.concat(frames, ignore_index=True))
    keep_columns = [
        "label",
        "device_id",
        "speed_percent",
        "rpm",
        "source_file",
        "record_index",
        "group_id",
        "window_id",
        "window_start",
        "window_end",
        *REDRAW_FEATURE_COLUMNS,
    ]
    return merged.loc[:, keep_columns]


def _build_window_bundle(row: dict[str, object], config: RedrawConfig) -> dict[str, object]:
    metadata = dict(row)
    record_index, window_index = _parse_window_id(str(metadata["window_id"]))
    metadata["record_index"] = record_index
    metadata["window_index"] = window_index
    metadata["window_start"] = int(metadata.get("window_start", window_index * config.window_step))
    metadata["window_end"] = int(metadata.get("window_end", metadata["window_start"] + config.window_size))

    raw = pd.read_csv(str(metadata["source_file"]))
    raw_signal = raw[str(record_index)].to_numpy(dtype=float)
    processed = _preprocess_signal(raw_signal, _to_visualization_config(config))
    start = window_index * config.window_step
    end = start + config.window_size
    if metadata["window_start"] != start or metadata["window_end"] != end:
        raise ValueError(
            "window metadata does not match window_id-derived slice: "
            f"{metadata['window_id']} -> {start}:{end}, table={metadata['window_start']}:{metadata['window_end']}"
        )
    window = processed[start:end]
    if window.shape[0] != config.window_size:
        raise ValueError(f"rebuilt window has {window.shape[0]} samples, expected {config.window_size}")

    time_axis = np.arange(window.shape[0], dtype=float) / config.processed_fs
    waveform_norm = normalize_window_waveform(window)
    spectrum_freqs, spectrum_amps = _hann_amplitude_spectrum(window, config.processed_fs)
    envelope_signal = _envelope_signal(window, _to_visualization_config(config))
    envelope_freqs, envelope_amps = _hann_amplitude_spectrum(envelope_signal, config.processed_fs)
    wavelet_ratios = _wavelet_packet_ratios(window, _to_visualization_config(config))
    low_mask = spectrum_freqs <= config.fft_low_max_hz
    low_norm = _normalize_trace(spectrum_amps[low_mask])
    full_mask = spectrum_freqs <= config.fft_full_max_hz
    full_db = _relative_db(spectrum_amps[full_mask], floor=FULL_SPECTRUM_DB_FLOOR)

    return {
        "metadata": metadata,
        "window": window,
        "time_axis": time_axis,
        "waveform_norm": waveform_norm,
        "spectrum_freqs": spectrum_freqs,
        "spectrum_amps": spectrum_amps,
        "low_freqs": spectrum_freqs[low_mask],
        "low_amplitudes": spectrum_amps[low_mask],
        "low_normalized": low_norm,
        "full_freqs": spectrum_freqs[full_mask],
        "full_amplitudes": spectrum_amps[full_mask],
        "full_relative_db": full_db,
        "envelope_signal": envelope_signal,
        "envelope_freqs": envelope_freqs,
        "envelope_amps": envelope_amps,
        "wavelet_ratios": wavelet_ratios,
    }


def _parse_window_id(window_id: str) -> tuple[int, int]:
    try:
        _, record_index, window_index = window_id.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"window_id does not match expected '<stem>_<record>_<window>' format: {window_id}") from exc
    return int(record_index), int(window_index)


def _normalize_trace(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    scale = float(np.max(array))
    if scale <= 0.0:
        return np.zeros_like(array)
    return array / scale


def _relative_db(values: np.ndarray, floor: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    reference = float(np.max(array))
    if reference <= 0.0:
        return np.full(array.shape, floor, dtype=float)
    relative = 20.0 * np.log10(np.maximum(array, _EPS) / reference)
    return np.clip(relative, floor, 0.0)


def _write_time_waveform_figure(bundles: dict[str, dict[str, object]], config: RedrawConfig) -> Path:
    rows = []
    fig, axes = _create_page("六类状态代表样本的归一化时域波形", config)
    for ax, label in zip(axes, CLASS_LAYOUT_ORDER):
        bundle = bundles[label]
        metadata = bundle["metadata"]
        ax.plot(bundle["time_axis"], bundle["waveform_norm"], color="#1f4e79", linewidth=1.1)
        ax.set_xlim(0.0, config.window_size / config.processed_fs)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("归一化幅值")
        ax.set_title(_panel_title(metadata), fontsize=config.panel_title_size)
        _style_axis(ax, config)
        rows.extend(
            {
                **_manifest_row(metadata),
                "sample_index": int(index),
                "time_s": float(time_s),
                "normalized_amplitude": float(amplitude),
                "processed_amplitude": float(raw_amplitude),
            }
            for index, (time_s, amplitude, raw_amplitude) in enumerate(
                zip(bundle["time_axis"], bundle["waveform_norm"], bundle["window"])
            )
        )

    csv_path = config.output_root / "fig4_2_time_waveform.csv"
    png_path = config.output_root / "fig4_2_time_waveform.png"
    pdf_path = config.output_root / "fig4_2_time_waveform.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=config.csv_encoding)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return csv_path


def _write_low_frequency_figure(bundles: dict[str, dict[str, object]], config: RedrawConfig) -> Path:
    rows = []
    fig, axes = _create_page("六类状态代表样本的低频频谱", config)
    for ax, label in zip(axes, CLASS_LAYOUT_ORDER):
        bundle = bundles[label]
        metadata = bundle["metadata"]
        ax.plot(bundle["low_freqs"], bundle["low_normalized"], color="#c55a11", linewidth=1.1)
        _annotate_harmonics(ax, float(metadata["rpm"]) / 60.0, config.fft_low_max_hz)
        ax.set_xlim(0.0, config.fft_low_max_hz)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("频率 / Hz")
        ax.set_ylabel("归一化谱幅值")
        ax.set_title(_panel_title(metadata), fontsize=config.panel_title_size)
        _style_axis(ax, config)
        rows.extend(
            {
                **_manifest_row(metadata),
                "freq_hz": float(freq),
                "amplitude": float(amplitude),
                "normalized_amplitude": float(norm_amp),
            }
            for freq, amplitude, norm_amp in zip(
                bundle["low_freqs"],
                bundle["low_amplitudes"],
                bundle["low_normalized"],
            )
        )

    csv_path = config.output_root / "fig4_3_low_frequency_spectrum.csv"
    png_path = config.output_root / "fig4_3_low_frequency_spectrum.png"
    pdf_path = config.output_root / "fig4_3_low_frequency_spectrum.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=config.csv_encoding)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return csv_path


def _write_full_spectrum_figure(bundles: dict[str, dict[str, object]], config: RedrawConfig) -> Path:
    rows = []
    fig, axes = _create_page("六类状态代表样本的0～5000 Hz全频频谱", config)
    for ax, label in zip(axes, CLASS_LAYOUT_ORDER):
        bundle = bundles[label]
        metadata = bundle["metadata"]
        _add_full_spectrum_bands(ax)
        ax.plot(bundle["full_freqs"], bundle["full_relative_db"], color="#2f5597", linewidth=1.1)
        ax.set_xlim(0.0, config.fft_full_max_hz)
        ax.set_ylim(FULL_SPECTRUM_DB_FLOOR, 0.0)
        ax.set_xlabel("频率 / Hz")
        ax.set_ylabel("相对幅值 / dB")
        ax.set_title(_panel_title(metadata), fontsize=config.panel_title_size)
        _style_axis(ax, config)
        rows.extend(
            {
                **_manifest_row(metadata),
                "freq_hz": float(freq),
                "amplitude": float(amplitude),
                "relative_db": float(relative_db),
            }
            for freq, amplitude, relative_db in zip(
                bundle["full_freqs"],
                bundle["full_amplitudes"],
                bundle["full_relative_db"],
            )
        )

    csv_path = config.output_root / "fig4_4_full_spectrum.csv"
    png_path = config.output_root / "fig4_4_full_spectrum.png"
    pdf_path = config.output_root / "fig4_4_full_spectrum.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=config.csv_encoding)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return csv_path


def _write_wavelet_figure(bundles: dict[str, dict[str, object]], config: RedrawConfig) -> Path:
    rows = []
    fig, axes = _create_page("六类状态代表样本的小波包节点能量占比", config)
    y_max = max(float(np.max(bundle["wavelet_ratios"])) for bundle in bundles.values())
    y_limit = min(1.0, max(0.2, y_max * 1.15))

    for ax, label in zip(axes, CLASS_LAYOUT_ORDER):
        bundle = bundles[label]
        metadata = bundle["metadata"]
        indices = np.arange(8)
        values = np.asarray(bundle["wavelet_ratios"], dtype=float)
        bars = ax.bar(indices, values, color="#5b9bd5", edgecolor="#2f5597", linewidth=0.7)
        ax.set_ylim(0.0, y_limit)
        ax.set_xticks(indices)
        ax.set_xlabel("节点编号")
        ax.set_ylabel("能量占比")
        ax.set_title(_panel_title(metadata), fontsize=config.panel_title_size)
        _style_axis(ax, config, y_grid_only=True)
        for bar, value in zip(bars, values):
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                float(value) + y_limit * 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=max(config.tick_label_size - 0.3, 8),
            )
        rows.extend(
            {
                **_manifest_row(metadata),
                "node_index": int(index),
                "energy_ratio": float(value),
            }
            for index, value in enumerate(values)
        )

    csv_path = config.output_root / "fig4_5_wavelet_packet_bar.csv"
    png_path = config.output_root / "fig4_5_wavelet_packet_bar.png"
    pdf_path = config.output_root / "fig4_5_wavelet_packet_bar.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=config.csv_encoding)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return csv_path


def _write_envelope_figure(bundles: dict[str, dict[str, object]], config: RedrawConfig) -> Path:
    dominant_peaks = [
        _dominant_peak_hz(bundle["envelope_freqs"], bundle["envelope_amps"], config.envelope_expanded_max_hz)
        for bundle in bundles.values()
    ]
    envelope_limit = (
        config.envelope_expanded_max_hz
        if should_expand_envelope_limit(dominant_peaks)
        else config.envelope_default_max_hz
    )

    rows = []
    fig, axes = _create_page("六类状态代表样本的包络谱", config)
    for ax, label in zip(axes, CLASS_LAYOUT_ORDER):
        bundle = bundles[label]
        metadata = bundle["metadata"]
        mask = bundle["envelope_freqs"] <= envelope_limit
        env_freqs = bundle["envelope_freqs"][mask]
        env_amps = bundle["envelope_amps"][mask]
        ax.plot(env_freqs, env_amps, color="#2f7d32", linewidth=1.1)
        _annotate_envelope_peaks(ax, env_freqs, env_amps)
        ax.set_xlim(0.0, envelope_limit)
        ax.set_ylim(0.0, max(float(np.max(env_amps)) * 1.08, _EPS))
        ax.set_xlabel("频率 / Hz")
        ax.set_ylabel("幅值 / a.u.")
        ax.set_title(_panel_title(metadata), fontsize=config.panel_title_size)
        _style_axis(ax, config)
        rows.extend(
            {
                **_manifest_row(metadata),
                "freq_hz": float(freq),
                "amplitude": float(amplitude),
                "envelope_limit_hz": float(envelope_limit),
            }
            for freq, amplitude in zip(env_freqs, env_amps)
        )

    csv_path = config.output_root / "fig4_6_envelope_spectrum.csv"
    png_path = config.output_root / "fig4_6_envelope_spectrum.png"
    pdf_path = config.output_root / "fig4_6_envelope_spectrum.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=config.csv_encoding)
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return csv_path


def _dominant_peak_hz(freqs: np.ndarray, amplitudes: np.ndarray, freq_max: float) -> float:
    mask = (freqs > 0.0) & (freqs <= freq_max)
    if not np.any(mask):
        return 0.0
    masked_freqs = freqs[mask]
    masked_amps = amplitudes[mask]
    if masked_amps.size == 0 or float(np.max(masked_amps)) <= 0.0:
        return 0.0
    return float(masked_freqs[int(np.argmax(masked_amps))])


def _annotate_harmonics(ax: plt.Axes, one_x_hz: float, freq_max: float) -> None:
    y_top = ax.get_ylim()[1]
    for multiple in (1, 2, 3):
        harmonic = one_x_hz * multiple
        if harmonic > freq_max:
            continue
        ax.axvline(harmonic, color="#7f6000", linestyle="--", linewidth=0.8, alpha=0.85)
        ax.text(harmonic, y_top * 0.98, f"{multiple}X", ha="center", va="top", color="#7f6000")


def _annotate_envelope_peaks(ax: plt.Axes, freqs: np.ndarray, amplitudes: np.ndarray) -> None:
    if freqs.size == 0 or amplitudes.size == 0:
        return
    valid_mask = freqs > 0.0
    if not np.any(valid_mask):
        return
    valid_freqs = freqs[valid_mask]
    valid_amps = amplitudes[valid_mask]
    main_index = int(np.argmax(valid_amps))
    main_freq = float(valid_freqs[main_index])
    main_amp = float(valid_amps[main_index])
    ax.plot([main_freq], [main_amp], marker="o", color="#1c5d1f", markersize=3)
    ax.text(main_freq, main_amp, f" 主峰 {main_freq:.1f} Hz", ha="left", va="bottom", color="#1c5d1f")

    modulation_mask = valid_freqs < max(main_freq * 0.5, 5.0)
    if np.any(modulation_mask):
        mod_freqs = valid_freqs[modulation_mask]
        mod_amps = valid_amps[modulation_mask]
        mod_index = int(np.argmax(mod_amps))
        mod_freq = float(mod_freqs[mod_index])
        mod_amp = float(mod_amps[mod_index])
        if mod_freq > 0.0 and not np.isclose(mod_freq, main_freq):
            ax.plot([mod_freq], [mod_amp], marker="s", color="#4f6228", markersize=3)
            ax.text(mod_freq, mod_amp, f" 调制 {mod_freq:.1f} Hz", ha="left", va="bottom", color="#4f6228")


def _add_full_spectrum_bands(ax: plt.Axes) -> None:
    for low, high, color in (
        (0.0, 500.0, "#f4f6f8"),
        (500.0, 2000.0, "#eef3f7"),
        (2000.0, 5000.0, "#f7f9fb"),
    ):
        ax.axvspan(low, high, color=color, alpha=1.0, zorder=0)


def _create_page(title: str, config: RedrawConfig | None = None) -> tuple[plt.Figure, list[plt.Axes]]:
    page_config = config or RedrawConfig()
    fig, axes = plt.subplots(3, 2, figsize=(16.0 / 2.54, 19.0 / 2.54))
    fig.suptitle(title, fontsize=page_config.page_title_size)
    return fig, list(axes.flatten())


def _style_axis(ax: plt.Axes, config: RedrawConfig, y_grid_only: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.xaxis.label.set_size(config.axis_label_size)
    ax.yaxis.label.set_size(config.axis_label_size)
    ax.tick_params(axis="both", labelsize=config.tick_label_size)
    if y_grid_only:
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    else:
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)


def _panel_title(metadata: dict[str, object]) -> str:
    return f"{metadata['label']} | {metadata['device_id']} | {int(float(metadata['rpm']))} rpm"


def _manifest_row(metadata: dict[str, object]) -> dict[str, object]:
    ordered = {column: metadata[column] for column in MANIFEST_REQUIRED_COLUMNS}
    extra_columns = [
        "record_index",
        "window_index",
        "window_start",
        "window_end",
        "distance_to_center",
    ]
    for column in extra_columns:
        if column in metadata:
            ordered[column] = metadata[column]
    return ordered


def _write_parameter_summary(config: RedrawConfig, bundles: dict[str, dict[str, object]]) -> Path:
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "source_feature_tables": [str(path) for path in (*MOTOR2_FEATURE_TABLES, MOTOR4_FEATURE_TABLE)],
        "class_layout_order": CLASS_LAYOUT_ORDER,
        "representative_windows": [_manifest_row(bundle["metadata"]) for bundle in bundles.values()],
        "speed_rpm_lookup": {str(key): value for key, value in {**RPM_BY_SPEED, 70: MOTOR4_RPM}.items()},
    }
    output = config.output_root / "processing_parameters.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _to_visualization_config(config: RedrawConfig) -> VisualizationConfig:
    return VisualizationConfig(
        output_root=config.output_root,
        original_fs=config.original_fs,
        processed_fs=config.processed_fs,
        resample_up=config.resample_up,
        resample_down=config.resample_down,
        bandpass_low=config.bandpass_low,
        bandpass_high=config.bandpass_high,
        filter_order=config.filter_order,
        fft_low_max_hz=config.fft_low_max_hz,
        fft_full_max_hz=config.fft_full_max_hz,
        envelope_band_low=config.envelope_band_low,
        envelope_band_high=config.envelope_band_high,
        envelope_max_hz=config.envelope_expanded_max_hz,
        wavelet=config.wavelet,
        wavelet_level=config.wavelet_level,
        plot_dpi=config.plot_dpi,
        font_size=config.font_size,
        csv_encoding=config.csv_encoding,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw unified six-class feature figures from representative windows.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    result = run_six_class_feature_redraw(RedrawConfig(output_root=Path(args.output_root)))
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
