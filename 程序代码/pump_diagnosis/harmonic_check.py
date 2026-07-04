from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import signal
from pandas.errors import ParserError

from config import PipelineConfig
from pump_diagnosis.features import amplitude_spectrum
from pump_diagnosis.metadata import ConditionCatalog, build_file_index
from pump_diagnosis.signal_processing import preprocess_run


def build_energy_harmonic_check(
    features: pd.DataFrame,
    file_index: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    report = features.merge(
        file_index.loc[:, ["file_path", "run_id", "rpm"]].rename(columns={"rpm": "speed_rpm_workbook"}),
        on=["file_path", "run_id"],
        how="left",
    )
    report = report.rename(columns={"rpm": "speed_rpm_current"})
    report["freq_1x_hz"] = report["speed_rpm_workbook"] / 60.0
    report["freq_2x_hz"] = report["freq_1x_hz"] * 2.0
    report["freq_3x_hz"] = report["freq_1x_hz"] * 3.0
    report["search_band_width_hz"] = 2.0 * config.harmonic_tolerance
    report["fft_bin_width_hz"] = config.processed_fs / config.window_size

    for energy_col in ("energy_1x", "energy_2x", "energy_3x"):
        report[f"file_{energy_col}_nunique"] = report.groupby("file_path")[energy_col].transform("nunique")
        report[f"label_{energy_col}_nunique"] = report.groupby("label")[energy_col].transform("nunique")
        report[f"file_{energy_col}_constant"] = report[f"file_{energy_col}_nunique"].eq(1)
        report[f"label_{energy_col}_constant"] = report[f"label_{energy_col}_nunique"].eq(1)

    ordered_columns = [
        "sample_id",
        "label",
        "file_path",
        "run_id",
        "window_index",
        "speed_rpm_current",
        "speed_rpm_workbook",
        "freq_1x_hz",
        "freq_2x_hz",
        "freq_3x_hz",
        "search_band_width_hz",
        "fft_bin_width_hz",
        "energy_1x",
        "energy_2x",
        "energy_3x",
        "file_energy_1x_nunique",
        "file_energy_2x_nunique",
        "file_energy_3x_nunique",
        "label_energy_1x_nunique",
        "label_energy_2x_nunique",
        "label_energy_3x_nunique",
        "file_energy_1x_constant",
        "file_energy_2x_constant",
        "file_energy_3x_constant",
        "label_energy_1x_constant",
        "label_energy_2x_constant",
        "label_energy_3x_constant",
    ]
    existing = [column for column in ordered_columns if column in report.columns]
    remaining = [column for column in report.columns if column not in existing]
    return report.loc[:, existing + remaining]


def build_energy_harmonic_check_recomputed(
    features: pd.DataFrame,
    file_index: pd.DataFrame,
    manifests: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    merged = features.merge(
        file_index.loc[:, ["file_path", "run_id", "machine_id", "speed_percent", "rpm"]].rename(
            columns={
                "rpm": "speed_rpm_workbook",
            }
        ),
        on=["file_path", "run_id"],
        how="left",
    )
    merged = merged.merge(
        manifests.loc[:, ["file_path", "run_id", "source_column"]],
        on=["file_path", "run_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    merged = merged.rename(columns={"rpm": "speed_rpm_current"})
    merged["freq_1x_hz"] = merged["speed_rpm_workbook"] / 60.0
    merged["freq_2x_hz"] = merged["freq_1x_hz"] * 2.0
    merged["freq_3x_hz"] = merged["freq_1x_hz"] * 3.0
    merged["search_band_width_hz"] = 2.0 * config.harmonic_tolerance
    merged["fft_bin_width_hz"] = config.processed_fs / config.window_size
    merged["bandpass_gain_1x"] = merged["freq_1x_hz"].apply(lambda freq: _bandpass_gain_at(freq, config))
    merged["bandpass_gain_1x_db"] = 20.0 * np.log10(merged["bandpass_gain_1x"].clip(lower=np.finfo(float).tiny))

    raw_windows = _recompute_rotation_windows(merged, config)
    merged = merged.merge(
        raw_windows,
        on=["sample_id", "file_path", "run_id", "window_index"],
        how="left",
        suffixes=("", "_recomputed"),
    )
    merged["rot_1x_amp"] = merged["rot_1x_amp_recomputed"]
    merged["rot_2x_amp"] = merged["rot_2x_amp_recomputed"]
    merged["rot_3x_amp"] = merged["rot_3x_amp_recomputed"]
    merged["energy_1x"] = merged["energy_1x_recomputed"]
    merged["energy_2x"] = merged["energy_2x_recomputed"]
    merged["energy_3x"] = merged["energy_3x_recomputed"]
    merged["rot_1x_peak_freq_hz"] = merged["rot_1x_peak_freq_hz_recomputed"]
    merged["rot_2x_peak_freq_hz"] = merged["rot_2x_peak_freq_hz_recomputed"]
    merged["rot_3x_peak_freq_hz"] = merged["rot_3x_peak_freq_hz_recomputed"]
    merged["rot_1x_peak_delta_hz"] = merged["rot_1x_peak_delta_hz_recomputed"]
    merged["rot_2x_peak_delta_hz"] = merged["rot_2x_peak_delta_hz_recomputed"]
    merged["rot_3x_peak_delta_hz"] = merged["rot_3x_peak_delta_hz_recomputed"]

    for energy_col in ("energy_1x", "energy_2x", "energy_3x"):
        merged[f"file_{energy_col}_nunique"] = merged.groupby("file_path")[energy_col].transform("nunique")
        merged[f"label_{energy_col}_nunique"] = merged.groupby("label")[energy_col].transform("nunique")
        merged[f"file_{energy_col}_constant"] = merged[f"file_{energy_col}_nunique"].eq(1)
        merged[f"label_{energy_col}_constant"] = merged[f"label_{energy_col}_nunique"].eq(1)

    for amp_col in ("rot_1x_amp", "rot_2x_amp", "rot_3x_amp"):
        merged[f"file_{amp_col}_nunique"] = merged.groupby("file_path")[amp_col].transform("nunique")
        merged[f"label_{amp_col}_nunique"] = merged.groupby("label")[amp_col].transform("nunique")
        merged[f"file_{amp_col}_constant"] = merged[f"file_{amp_col}_nunique"].eq(1)
        merged[f"label_{amp_col}_constant"] = merged[f"label_{amp_col}_nunique"].eq(1)

    merged["theory_peak_match_1x"] = (merged["rot_1x_peak_delta_hz"].abs() <= merged["fft_bin_width_hz"]).fillna(False)
    merged["theory_peak_match_2x"] = (merged["rot_2x_peak_delta_hz"].abs() <= merged["fft_bin_width_hz"]).fillna(False)
    merged["theory_peak_match_3x"] = (merged["rot_3x_peak_delta_hz"].abs() <= merged["fft_bin_width_hz"]).fillna(False)

    ordered_columns = [
        "sample_id",
        "label",
        "file_path",
        "run_id",
        "source_column",
        "window_index",
        "machine_id",
        "speed_percent",
        "speed_rpm_current",
        "speed_rpm_workbook",
        "freq_1x_hz",
        "freq_2x_hz",
        "freq_3x_hz",
        "bandpass_gain_1x",
        "bandpass_gain_1x_db",
        "search_band_width_hz",
        "fft_bin_width_hz",
        "peak_frequency",
        "rot_1x_peak_freq_hz",
        "rot_2x_peak_freq_hz",
        "rot_3x_peak_freq_hz",
        "rot_1x_peak_delta_hz",
        "rot_2x_peak_delta_hz",
        "rot_3x_peak_delta_hz",
        "theory_peak_match_1x",
        "theory_peak_match_2x",
        "theory_peak_match_3x",
        "rot_1x_amp",
        "rot_2x_amp",
        "rot_3x_amp",
        "energy_1x",
        "energy_2x",
        "energy_3x",
        "file_rot_1x_amp_nunique",
        "file_rot_2x_amp_nunique",
        "file_rot_3x_amp_nunique",
        "label_rot_1x_amp_nunique",
        "label_rot_2x_amp_nunique",
        "label_rot_3x_amp_nunique",
        "file_energy_1x_nunique",
        "file_energy_2x_nunique",
        "file_energy_3x_nunique",
        "label_energy_1x_nunique",
        "label_energy_2x_nunique",
        "label_energy_3x_nunique",
        "file_rot_1x_amp_constant",
        "file_rot_2x_amp_constant",
        "file_rot_3x_amp_constant",
        "label_rot_1x_amp_constant",
        "label_rot_2x_amp_constant",
        "label_rot_3x_amp_constant",
        "file_energy_1x_constant",
        "file_energy_2x_constant",
        "file_energy_3x_constant",
        "label_energy_1x_constant",
        "label_energy_2x_constant",
        "label_energy_3x_constant",
    ]
    existing = [column for column in ordered_columns if column in merged.columns]
    remaining = [column for column in merged.columns if column not in existing]
    return merged.loc[:, existing + remaining]


def run_energy_harmonic_check(
    feature_root: Path,
    config: PipelineConfig,
    *,
    recompute_rotation_features: bool = False,
) -> pd.DataFrame:
    config.output_root.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(feature_root / "train_features_raw.csv")
    test = pd.read_csv(feature_root / "test_features_raw.csv")
    features = pd.concat([train, test], ignore_index=True)
    catalog = ConditionCatalog.from_workbook(config.condition_workbook)
    file_index = build_file_index(config, catalog=catalog)
    if recompute_rotation_features:
        train_manifest = pd.read_csv(feature_root / "train_manifest.csv")
        test_manifest = pd.read_csv(feature_root / "test_manifest.csv")
        manifests = pd.concat([train_manifest, test_manifest], ignore_index=True)
        report = build_energy_harmonic_check_recomputed(features, file_index, manifests, config)
    else:
        report = build_energy_harmonic_check(features, file_index, config)
    report.to_csv(config.output_root / "energy_harmonic_check.csv", index=False)
    return report


def _recompute_rotation_windows(features: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    tasks: list[dict[str, object]] = []
    grouped_features = features.sort_values(["run_id", "window_index"]).groupby("run_id", sort=False)

    for run_id, run_features in grouped_features:
        first = run_features.iloc[0]
        tasks.append(
            {
                "file_path": str(first["file_path"]),
                "source_column": str(first["source_column"]),
                "rpm": float(first["speed_rpm_workbook"]) if pd.notna(first["speed_rpm_workbook"]) else np.nan,
                "windows": [
                    {
                        "sample_id": row.sample_id,
                        "file_path": row.file_path,
                        "run_id": row.run_id,
                        "window_index": row.window_index,
                        "window_start": float(row.window_start),
                        "window_end": float(row.window_end),
                    }
                    for row in run_features.itertuples(index=False)
                ],
            }
        )

    rows: list[dict[str, object]] = []
    worker_count = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_process_rotation_run, task, config) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result().to_dict("records"))
            if completed % 100 == 0:
                print(f"rotation windows recomputed: {completed}/{len(futures)}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["run_id", "window_index", "sample_id"]).reset_index(drop=True)
    return frame


def _process_rotation_run(task: dict[str, object], config: PipelineConfig) -> pd.DataFrame:
    file_path = str(task["file_path"])
    source_column = str(task["source_column"])
    rpm = float(task["rpm"])
    windows = task["windows"]

    try:
        raw_frame = pd.read_csv(file_path, low_memory=False)
    except ParserError:
        raw_frame = pd.read_csv(file_path, engine="python")
    raw = raw_frame[source_column].to_numpy(dtype=np.float64)
    processed = preprocess_run(raw, config)

    rows: list[dict[str, object]] = []
    for window in windows:
        start = int(round(float(window["window_start"]) * config.processed_fs))
        end = int(round(float(window["window_end"]) * config.processed_fs))
        metrics = _rotation_window_metrics(processed[start:end], rpm, config)
        metrics.update(
            {
                "sample_id": window["sample_id"],
                "file_path": window["file_path"],
                "run_id": window["run_id"],
                "window_index": window["window_index"],
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _rotation_window_metrics(window: np.ndarray, rpm: float, config: PipelineConfig) -> dict[str, object]:
    freqs, amps = amplitude_spectrum(window, config.processed_fs)
    power = np.square(amps)
    base_freq = rpm / 60.0
    output: dict[str, object] = {}
    for multiple in (1, 2, 3):
        target_freq = base_freq * multiple
        amp = _sample_at(freqs, amps, target_freq)
        energy = _band_energy(freqs, power, target_freq - config.harmonic_tolerance, target_freq + config.harmonic_tolerance)
        band_mask = (freqs >= target_freq - config.harmonic_tolerance) & (freqs <= target_freq + config.harmonic_tolerance)
        if np.any(band_mask):
            local_freqs = freqs[band_mask]
            local_amps = amps[band_mask]
            local_idx = int(np.argmax(local_amps))
            peak_freq = float(local_freqs[local_idx])
            peak_delta = float(peak_freq - target_freq)
        else:
            peak_freq = np.nan
            peak_delta = np.nan
        output[f"rot_{multiple}x_amp_recomputed"] = float(amp)
        output[f"energy_{multiple}x_recomputed"] = float(energy)
        output[f"rot_{multiple}x_peak_freq_hz_recomputed"] = peak_freq
        output[f"rot_{multiple}x_peak_delta_hz_recomputed"] = peak_delta
    return output


def _band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _sample_at(freqs: np.ndarray, amps: np.ndarray, target: float) -> float:
    if freqs.size == 0:
        return 0.0
    index = int(np.argmin(np.abs(freqs - target)))
    return float(amps[index])


def _bandpass_gain_at(freq_hz: float, config: PipelineConfig) -> float:
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.bandpass_low, config.bandpass_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    _, response = signal.sosfreqz(sos, worN=[freq_hz], fs=config.processed_fs)
    return float(np.abs(response[0]))
