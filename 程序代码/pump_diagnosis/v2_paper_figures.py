from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pump_diagnosis.fault_feature_visualization import (
    VisualizationConfig,
    _configure_matplotlib,
    _envelope_signal,
    _hann_amplitude_spectrum,
    _preprocess_signal,
    _save_dual,
    _wavelet_packet_ratios,
)
from pump_diagnosis.three_class_feature_table import FEATURE_COLUMNS
from pump_diagnosis.unified_six_class_exploration_v2 import (
    DEFAULT_OUTPUT_ROOT as V2_OUTPUT_ROOT,
    RPM_BY_SPEED,
    MOTOR4_RPM,
    add_speed_metadata,
)


RESULTS_ROOT = V2_OUTPUT_ROOT.parent
DEFAULT_FIGURE_ROOT = RESULTS_ROOT / "v2_paper_figures"
MOTOR2_TABLES = {
    50: RESULTS_ROOT / "Motor2_50_740rpm_三分类_通道4_特征表.csv",
    75: RESULTS_ROOT / "Motor2_75_1110rpm_三分类_通道4_特征表.csv",
    100: RESULTS_ROOT / "Motor2_100_1480rpm_三分类_通道4_特征表.csv",
}
MOTOR4_TABLE = RESULTS_ROOT / "Motor4_70_2070rpm_四分类_通道4_特征表.csv"
CLASS_ORDER = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]


@dataclass(frozen=True)
class PaperFigureConfig:
    output_root: Path = DEFAULT_FIGURE_ROOT
    csv_encoding: str = "utf-8-sig"
    plot_dpi: int = 300
    processed_fs: int = 12_000
    waveform_seconds: float = 1.0
    fft_low_max_hz: float = 300.0
    fft_full_max_hz: float = 5000.0
    envelope_max_hz: float = 1000.0
    plot_config: VisualizationConfig = VisualizationConfig(
        output_root=DEFAULT_FIGURE_ROOT,
        bandpass_low=10.0,
        bandpass_high=5000.0,
        envelope_band_low=2000.0,
        envelope_band_high=5000.0,
    )


def aggregate_class_distribution(balance_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        balance_frame.groupby("label", as_index=False)
        .agg(window_count_total=("window_count_total", "sum"), group_count_total=("group_count_total", "sum"))
        .reset_index(drop=True)
    )
    order_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    return summary.sort_values(by="label", key=lambda series: series.map(order_map).fillna(len(order_map))).reset_index(drop=True)


def select_representative_groups(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, object]]:
    grouped = (
        frame.groupby(["label", "group_id"], as_index=False)
        .agg(
            device_id=("device_id", "first"),
            speed_percent=("speed_percent", "first"),
            rpm=("rpm", "first"),
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
            "device_id": row["device_id"],
            "speed_percent": int(row["speed_percent"]),
            "rpm": float(row["rpm"]),
            "group_id": row["group_id"],
            "source_file": row["source_file"],
            "record_index": int(row["record_index"]),
            "distance_to_center": float(distances[best_index]),
        }
    return selected


def run_v2_paper_figures(config: PaperFigureConfig) -> dict[str, object]:
    _configure_matplotlib(config.plot_config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "csv").mkdir(parents=True, exist_ok=True)

    motor2_frame = _load_motor2_frame()
    motor4_frame = _load_motor4_frame()
    selected = {
        "Motor-2": select_representative_groups(motor2_frame, FEATURE_COLUMNS),
        "Motor-4": select_representative_groups(motor4_frame, FEATURE_COLUMNS),
    }
    bundles = _build_device_bundles(selected, config)
    figure_paths = {
        "fig4_1": _plot_class_distribution(config),
        "fig4_2": _plot_time_waveform_grid(bundles, config),
        "fig4_3": _plot_fft_grid(bundles, config),
        "fig4_4": _plot_wavelet_grid(bundles, config),
        "fig4_5": _plot_envelope_grid(bundles, config),
        "fig4_6": _plot_model_metric_comparison(config),
        "fig4_7": _plot_bp_confusion_heatmap(config),
        "fig4_8": _plot_window_group_comparison(config),
    }

    representative_rows = []
    for device_id, mapping in selected.items():
        for label, row in mapping.items():
            representative_rows.append({"device_id": device_id, **row})
    pd.DataFrame(representative_rows).to_csv(
        config.output_root / "csv" / "representative_samples.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return {key: str(value) for key, value in figure_paths.items()}


def _load_motor2_frame() -> pd.DataFrame:
    frames = []
    for path in MOTOR2_TABLES.values():
        frame = pd.read_csv(path).copy()
        frame["device_id"] = "Motor-2"
        frames.append(frame)
    return add_speed_metadata(pd.concat(frames, ignore_index=True))


def _load_motor4_frame() -> pd.DataFrame:
    frame = pd.read_csv(MOTOR4_TABLE).copy()
    frame["device_id"] = "Motor-4"
    return add_speed_metadata(frame)


def _build_device_bundles(selected: dict[str, dict[str, dict[str, object]]], config: PaperFigureConfig) -> dict[str, dict[str, dict[str, object]]]:
    bundles: dict[str, dict[str, dict[str, object]]] = {}
    for device_id, label_map in selected.items():
        bundles[device_id] = {}
        for label, row in label_map.items():
            raw = pd.read_csv(row["source_file"])
            raw_signal = raw[str(row["record_index"])].to_numpy(dtype=float)
            processed = _preprocess_signal(raw_signal, config.plot_config)
            waveform_samples = int(config.waveform_seconds * config.processed_fs)
            waveform = processed[:waveform_samples]
            waveform_time = np.arange(waveform.shape[0], dtype=float) / config.processed_fs
            fft_freqs, fft_amps = _hann_amplitude_spectrum(processed, config.processed_fs)
            env_signal = _envelope_signal(processed, config.plot_config)
            env_freqs, env_amps = _hann_amplitude_spectrum(env_signal, config.processed_fs)
            wavelet_ratios = _wavelet_packet_ratios(processed, config.plot_config)
            bundles[device_id][label] = {
                "sample": row,
                "waveform_time": waveform_time,
                "waveform": waveform,
                "fft_freqs": fft_freqs,
                "fft_amps": fft_amps,
                "env_freqs": env_freqs,
                "env_amps": env_amps,
                "wavelet_ratios": wavelet_ratios,
            }
    return bundles


def _plot_class_distribution(config: PaperFigureConfig) -> Path:
    balance = pd.read_csv(V2_OUTPUT_ROOT / "sample_balance_report.csv")
    summary = aggregate_class_distribution(balance)
    summary.to_csv(config.output_root / "csv" / "fig4_1_class_distribution.csv", index=False, encoding=config.csv_encoding)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    color = "#4F81BD"
    axes[0].bar(summary["label"], summary["window_count_total"], color=color)
    axes[0].set_title("各类别窗口数量")
    axes[0].set_ylabel("窗口数")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(summary["label"], summary["group_count_total"], color="#9BBB59")
    axes[1].set_title("各类别采集段数量")
    axes[1].set_ylabel("采集段数")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("多转速统一六分类实验样本数量分布")
    return _save_figure(fig, config.output_root / "fig4_1_class_distribution", config)


def _plot_time_waveform_grid(bundles: dict[str, dict[str, dict[str, object]]], config: PaperFigureConfig) -> Path:
    rows = [("Motor-2", ["正常", "松动", "轴承故障"]), ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"])]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6), sharex=False)
    _hide_all_axes(axes)
    for row_idx, (device_id, labels) in enumerate(rows):
        y_values = np.concatenate([bundles[device_id][label]["waveform"] for label in labels])
        y_limits = (float(np.min(y_values)), float(np.max(y_values)))
        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)
            bundle = bundles[device_id][label]
            ax.plot(bundle["waveform_time"], bundle["waveform"], color="#1f4e79", linewidth=0.9)
            ax.set_xlim(0.0, config.waveform_seconds)
            ax.set_ylim(y_limits)
            ax.set_title(_sample_title(label, bundle["sample"]))
            ax.set_xlabel("时间 / s")
            ax.set_ylabel("幅值 / a.u.")
            ax.grid(True, alpha=0.2)
    fig.suptitle("代表样本时域波形对比")
    rows_long = []
    for device_id, labels in rows:
        for label in labels:
            bundle = bundles[device_id][label]
            frame = pd.DataFrame({"device_id": device_id, "label": label, "time_s": bundle["waveform_time"], "amplitude": bundle["waveform"]})
            rows_long.append(frame)
    pd.concat(rows_long, ignore_index=True).to_csv(
        config.output_root / "csv" / "fig4_2_time_waveforms.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return _save_figure(fig, config.output_root / "fig4_2_time_waveforms", config)


def _plot_fft_grid(bundles: dict[str, dict[str, dict[str, object]]], config: PaperFigureConfig) -> Path:
    rows = [
        ("Motor-2", ["正常", "松动", "轴承故障"], config.fft_low_max_hz),
        ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"], config.fft_low_max_hz),
        ("Motor-2", ["正常", "松动", "轴承故障"], config.fft_full_max_hz),
        ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"], config.fft_full_max_hz),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11.2), sharex=False)
    _hide_all_axes(axes)
    csv_rows = []
    for row_idx, (device_id, labels, freq_max) in enumerate(rows):
        y_max = max(float(np.max(bundles[device_id][label]["fft_amps"][bundles[device_id][label]["fft_freqs"] <= freq_max])) for label in labels)
        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)
            bundle = bundles[device_id][label]
            mask = bundle["fft_freqs"] <= freq_max
            ax.plot(bundle["fft_freqs"][mask], bundle["fft_amps"][mask], color="#c55a11", linewidth=0.9)
            _annotate_harmonics(ax, bundle["sample"]["rpm"] / 60.0, freq_max, y_max * 1.02)
            ax.set_xlim(0.0, freq_max)
            ax.set_ylim(0.0, y_max * 1.05)
            suffix = "低频" if freq_max <= config.fft_low_max_hz else "全频"
            ax.set_title(f"{_sample_title(label, bundle['sample'])}\n{suffix}")
            ax.set_xlabel("频率 / Hz")
            ax.set_ylabel("幅值 / a.u.")
            ax.grid(True, alpha=0.2)
            csv_rows.append(
                pd.DataFrame(
                    {
                        "device_id": device_id,
                        "label": label,
                        "range": suffix,
                        "frequency_hz": bundle["fft_freqs"][mask],
                        "amplitude": bundle["fft_amps"][mask],
                    }
                )
            )
    fig.suptitle("代表样本频谱对比（低频与全频）")
    pd.concat(csv_rows, ignore_index=True).to_csv(
        config.output_root / "csv" / "fig4_3_fft.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return _save_figure(fig, config.output_root / "fig4_3_fft", config)


def _plot_wavelet_grid(bundles: dict[str, dict[str, dict[str, object]]], config: PaperFigureConfig) -> Path:
    fig = plt.figure(figsize=(12, 10.0))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.1])
    rows = [("Motor-2", ["正常", "松动", "轴承故障"]), ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"])]
    csv_rows = []
    for row_idx, (device_id, labels) in enumerate(rows):
        for col_idx, label in enumerate(labels):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            bundle = bundles[device_id][label]
            nodes = [f"节点{i}" for i in range(8)]
            ax.bar(nodes, bundle["wavelet_ratios"], color="#5b9bd5", edgecolor="#2f5597")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(_sample_title(label, bundle["sample"]))
            ax.tick_params(axis="x", rotation=25)
            ax.set_ylabel("能量占比")
            ax.grid(True, axis="y", alpha=0.2)
            csv_rows.append(
                pd.DataFrame(
                    {
                        "device_id": device_id,
                        "label": label,
                        "node": nodes,
                        "energy_ratio": bundle["wavelet_ratios"],
                    }
                )
            )
    for heat_idx, (device_id, labels, slice_spec) in enumerate(
        [("Motor-2", ["正常", "松动", "轴承故障"], (2, slice(0, 2))), ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"], (2, slice(2, 4)))]
    ):
        ax = fig.add_subplot(grid[slice_spec[0], slice_spec[1]])
        heatmap = pd.DataFrame(
            {f"节点{i}": [float(bundles[device_id][label]["wavelet_ratios"][i]) for label in labels] for i in range(8)},
            index=labels,
        )
        sns.heatmap(heatmap, ax=ax, cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar=heat_idx == 1, annot=True, fmt=".2f")
        ax.set_title(f"{device_id} 小波包热力图")
        ax.set_xlabel("频带节点")
        ax.set_ylabel("类别")
        csv_rows.append(heatmap.reset_index(names="label").assign(device_id=device_id))
    fig.suptitle("代表样本小波包能量分布")
    pd.concat(csv_rows, ignore_index=True).to_csv(
        config.output_root / "csv" / "fig4_4_wavelet.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return _save_figure(fig, config.output_root / "fig4_4_wavelet", config)


def _plot_envelope_grid(bundles: dict[str, dict[str, dict[str, object]]], config: PaperFigureConfig) -> Path:
    rows = [("Motor-2", ["正常", "松动", "轴承故障"]), ("Motor-4", ["正常", "转子不平衡", "联轴器不对中", "汽蚀"])]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6), sharex=False)
    _hide_all_axes(axes)
    csv_rows = []
    for row_idx, (device_id, labels) in enumerate(rows):
        y_max = max(float(np.max(bundles[device_id][label]["env_amps"][bundles[device_id][label]["env_freqs"] <= config.envelope_max_hz])) for label in labels)
        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)
            bundle = bundles[device_id][label]
            mask = bundle["env_freqs"] <= config.envelope_max_hz
            ax.plot(bundle["env_freqs"][mask], bundle["env_amps"][mask], color="#2f7d32", linewidth=0.9)
            _annotate_harmonics(ax, bundle["sample"]["rpm"] / 60.0, config.envelope_max_hz, y_max * 1.02)
            ax.set_xlim(0.0, config.envelope_max_hz)
            ax.set_ylim(0.0, y_max * 1.05)
            ax.set_title(_sample_title(label, bundle["sample"]))
            ax.set_xlabel("频率 / Hz")
            ax.set_ylabel("幅值 / a.u.")
            ax.grid(True, alpha=0.2)
            csv_rows.append(
                pd.DataFrame(
                    {
                        "device_id": device_id,
                        "label": label,
                        "frequency_hz": bundle["env_freqs"][mask],
                        "amplitude": bundle["env_amps"][mask],
                    }
                )
            )
    fig.suptitle("代表样本包络谱对比")
    pd.concat(csv_rows, ignore_index=True).to_csv(
        config.output_root / "csv" / "fig4_5_envelope.csv",
        index=False,
        encoding=config.csv_encoding,
    )
    return _save_figure(fig, config.output_root / "fig4_5_envelope", config)


def _plot_model_metric_comparison(config: PaperFigureConfig) -> Path:
    frame = pd.read_csv(V2_OUTPUT_ROOT / "six_class_models" / "model_summary.csv")
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    long = frame.loc[:, ["model", *metrics]].melt(id_vars="model", var_name="metric", value_name="value")
    long.to_csv(config.output_root / "csv" / "fig4_6_model_metrics.csv", index=False, encoding=config.csv_encoding)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sns.barplot(data=long, x="metric", y="value", hue="model", ax=ax, palette="deep")
    ax.set_ylim(0.75, 1.02)
    ax.set_xlabel("指标")
    ax.set_ylabel("数值")
    ax.set_title("三种模型窗口级指标对比")
    ax.grid(True, axis="y", alpha=0.2)
    return _save_figure(fig, config.output_root / "fig4_6_model_metrics", config)


def _plot_bp_confusion_heatmap(config: PaperFigureConfig) -> Path:
    frame = pd.read_csv(V2_OUTPUT_ROOT / "six_class_models" / "bp" / "confusion_matrix.csv", index_col=0)
    frame.to_csv(config.output_root / "csv" / "fig4_7_bp_confusion_matrix.csv", encoding=config.csv_encoding)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("BP神经网络测试集混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    return _save_figure(fig, config.output_root / "fig4_7_bp_confusion_matrix", config)


def _plot_window_group_comparison(config: PaperFigureConfig) -> Path:
    rows = []
    for model in ["bp", "random_forest", "svm"]:
        metrics = pd.read_csv(V2_OUTPUT_ROOT / "six_class_models" / model / "test_metrics.csv").iloc[0]
        rows.extend(
            [
                {"model": model, "decision_level": "窗口级", "accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"])},
                {"model": model, "decision_level": "采集段-多数投票", "accuracy": float(metrics["group_majority_accuracy"]), "macro_f1": float(metrics["group_majority_macro_f1"])},
                {"model": model, "decision_level": "采集段-平均概率", "accuracy": float(metrics["group_probability_accuracy"]), "macro_f1": float(metrics["group_probability_macro_f1"])},
            ]
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(config.output_root / "csv" / "fig4_8_window_group_comparison.csv", index=False, encoding=config.csv_encoding)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.barplot(data=frame, x="model", y="accuracy", hue="decision_level", ax=axes[0], palette="deep")
    sns.barplot(data=frame, x="model", y="macro_f1", hue="decision_level", ax=axes[1], palette="deep")
    axes[0].set_ylim(0.8, 1.02)
    axes[1].set_ylim(0.8, 1.02)
    axes[0].set_title("Accuracy 对比")
    axes[1].set_title("Macro-F1 对比")
    for ax in axes:
        ax.set_xlabel("模型")
        ax.set_ylabel("数值")
        ax.grid(True, axis="y", alpha=0.2)
    return _save_figure(fig, config.output_root / "fig4_8_window_group_comparison", config)


def _annotate_harmonics(ax: plt.Axes, one_x_hz: float, freq_max: float, y_max: float) -> None:
    for multiple in (1, 2, 3):
        harmonic = one_x_hz * multiple
        if harmonic > freq_max:
            continue
        ax.axvline(harmonic, color="#7f6000", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(harmonic, y_max * 0.96, f"{multiple}X", rotation=90, va="top", ha="center", color="#7f6000", fontsize=8)


def _sample_title(label: str, sample: dict[str, object]) -> str:
    return f"{label}\n({int(sample['speed_percent'])}%, {int(sample['rpm'])} r/min)"


def _hide_all_axes(axes) -> None:
    axes_array = np.asarray(axes)
    for ax in axes_array.ravel():
        ax.set_visible(False)


def _save_figure(fig: plt.Figure, base_path: Path, config: PaperFigureConfig) -> Path:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    _save_dual(fig, png_path, pdf_path, config.plot_dpi)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready V2 figures.")
    parser.add_argument("--output-root", default=str(DEFAULT_FIGURE_ROOT))
    args = parser.parse_args()
    result = run_v2_paper_figures(PaperFigureConfig(output_root=Path(args.output_root)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
