from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig


def generate_stage1_plots(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    feature_csv: Path,
    config: PipelineConfig,
) -> dict[str, Path]:
    output_dir = config.output_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "class_counts": output_dir / "class_counts.png",
        "filter_response": output_dir / "filter_response.png",
        "preprocessing_spectra": output_dir / "preprocessing_spectra.png",
        "wavelet_energy": output_dir / "wavelet_energy.png",
    }
    _plot_class_counts(train_manifest, test_manifest, paths["class_counts"], config.plot_dpi)
    _plot_filter_response(paths["filter_response"], config.plot_dpi)
    _plot_preprocessing_spectra(feature_csv, paths["preprocessing_spectra"], config.plot_dpi)
    _plot_wavelet_energy(feature_csv, paths["wavelet_energy"], config.plot_dpi)
    return paths


def _plot_class_counts(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    labels = sorted(set(train_manifest["label"]).union(test_manifest["label"]))
    train_counts = train_manifest["label"].value_counts().reindex(labels, fill_value=0)
    test_counts = test_manifest["label"].value_counts().reindex(labels, fill_value=0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, train_counts.values, width=0.4, label="train")
    ax.bar(x + 0.2, test_counts.values, width=0.4, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_title("Class Counts")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_filter_response(output: Path, dpi: int) -> None:
    freqs = np.linspace(0, 6000, 400)
    response = np.where((freqs >= 10) & (freqs <= 5000), 1.0, 0.05)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, response)
    ax.set_title("Bandpass Response")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Gain")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_preprocessing_spectra(feature_csv: Path, output: Path, dpi: int) -> None:
    frame = pd.read_csv(feature_csv)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["peak_frequency"], marker="o")
    ax.set_title("Peak Frequency per Window")
    ax.set_xlabel("Window")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_wavelet_energy(feature_csv: Path, output: Path, dpi: int) -> None:
    frame = pd.read_csv(feature_csv)
    columns = [name for name in frame.columns if name.startswith("wp_energy_")]
    means = frame[columns].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(columns, means.values)
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_title("Wavelet Energy Ratios")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
