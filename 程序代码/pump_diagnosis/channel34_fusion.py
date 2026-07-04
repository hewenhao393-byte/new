from __future__ import annotations

import json
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ModelingConfig, PipelineConfig
from pump_diagnosis.corrected_harmonics import (
    fit_scaler_and_transform,
    remove_invalid_features,
    select_correlated_features,
    tune_and_evaluate_model,
)
from pump_diagnosis.features import amplitude_spectrum, extract_candidate_features
from pump_diagnosis.pipeline import METADATA_COLUMNS
from pump_diagnosis.signal_processing import iter_windows, preprocess_run


CHANNEL3_FEATURE_NAMES = [
    "rot_1x_amp",
    "rot_2x_amp",
    "rot_3x_amp",
    "energy_1x",
    "energy_2x",
    "energy_3x",
    "amp_2x_div_1x",
    "amp_3x_div_1x",
    "harmonic_energy_1x_3x",
    "rms",
    "std",
    "peak_to_peak",
    "crest_factor",
    "impulse_factor",
    "margin_factor",
    "clearance_factor",
    "kurtosis",
    "spectral_centroid",
    "spectral_entropy",
    "low_frequency_energy_ratio",
    "mid_frequency_energy_ratio",
    "wavelet_packet_energy_entropy",
    "envelope_rms",
    "envelope_kurtosis",
    "envelope_spectral_entropy",
    "envelope_spectral_peak",
]

CHANNEL4_FEATURE_NAMES = [
    "rot_1x_amp",
    "energy_1x",
    "rms",
    "dominant_frequency",
    "spectral_centroid",
    "rot_2x_amp",
    "rot_3x_amp",
    "energy_2x",
    "energy_3x",
    "amp_2x_div_1x",
    "amp_3x_div_1x",
    "peak_to_peak",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "kurtosis",
    "peak_interval_cv",
    "autocorrelation_peak",
    "envelope_rms",
    "envelope_kurtosis",
    "envelope_spectral_peak",
    "envelope_spectral_entropy",
    "high_frequency_energy_ratio",
    "2000_5000_energy_ratio",
    "wavelet_packet_energy_entropy",
    "wp_high_frequency_energy_ratio",
    "wavelet_packet_high_frequency_energy",
    "envelope_high_frequency_energy_ratio",
    "spectral_entropy",
    "low_frequency_energy_ratio",
]

CHANNEL34_METADATA_COLUMNS = METADATA_COLUMNS + ["split", "source_file_base"]
CHANNEL34_EXTENDED_METADATA_COLUMNS = CHANNEL34_METADATA_COLUMNS + [
    "source_file",
    "speed_percent",
    "raw_fault",
    "operating_condition",
]

CHANNEL3_PRIORITY_FEATURES = {
    "ch3_rot_1x_amp",
    "ch3_rot_2x_amp",
    "ch3_rot_3x_amp",
    "ch3_energy_1x",
    "ch3_energy_2x",
    "ch3_energy_3x",
    "ch3_amp_2x_div_1x",
    "ch3_amp_3x_div_1x",
    "ch3_harmonic_energy_1x_3x",
    "ch3_rms",
    "ch3_std",
    "ch3_peak_to_peak",
    "ch3_crest_factor",
    "ch3_impulse_factor",
    "ch3_margin_factor",
    "ch3_clearance_factor",
    "ch3_kurtosis",
    "ch3_spectral_centroid",
    "ch3_spectral_entropy",
    "ch3_low_frequency_energy_ratio",
    "ch3_mid_frequency_energy_ratio",
    "ch3_wavelet_packet_energy_entropy",
    "ch3_envelope_rms",
    "ch3_envelope_kurtosis",
    "ch3_envelope_spectral_entropy",
    "ch3_envelope_spectral_peak",
}

CHANNEL4_PRIORITY_FEATURES = {
    "ch4_rot_1x_amp",
    "ch4_energy_1x",
    "ch4_rms",
    "ch4_dominant_frequency",
    "ch4_spectral_centroid",
    "ch4_rot_2x_amp",
    "ch4_rot_3x_amp",
    "ch4_energy_2x",
    "ch4_energy_3x",
    "ch4_amp_2x_div_1x",
    "ch4_amp_3x_div_1x",
    "ch4_peak_to_peak",
    "ch4_crest_factor",
    "ch4_impulse_factor",
    "ch4_clearance_factor",
    "ch4_kurtosis",
    "ch4_peak_interval_cv",
    "ch4_autocorrelation_peak",
    "ch4_envelope_rms",
    "ch4_envelope_kurtosis",
    "ch4_envelope_spectral_peak",
    "ch4_envelope_spectral_entropy",
    "ch4_high_frequency_energy_ratio",
    "ch4_2000_5000_energy_ratio",
    "ch4_wavelet_packet_energy_entropy",
    "ch4_wp_high_frequency_energy_ratio",
    "ch4_wavelet_packet_high_frequency_energy",
    "ch4_envelope_high_frequency_energy_ratio",
    "ch4_spectral_entropy",
    "ch4_low_frequency_energy_ratio",
}

FUSION_PRIORITY_FEATURES = CHANNEL3_PRIORITY_FEATURES | CHANNEL4_PRIORITY_FEATURES

BALANCE_MODES = ("unbalanced", "balanced_min", "balanced_1p5min")
BALANCE_KEY_COLUMNS = ["split", "run_id", "source_file_base", "window_start", "window_end", "label"]
BALANCE_STRATA_COLUMNS = [
    "class_name",
    "source_file",
    "motor_id",
    "speed_percent",
    "speed_rpm",
    "operating_condition",
]


@dataclass(frozen=True)
class Channel34Outputs:
    output_root: Path
    channel3_raw: Path
    channel4_raw: Path
    channel3_aligned: Path
    channel4_aligned: Path
    fused_raw: Path


def source_file_base_from_path(path: Path | str) -> str:
    stem = Path(path).stem
    return re.sub(r"-通道\d+$", "", stem)


def load_split_manifests(feature_root: Path) -> dict[str, pd.DataFrame]:
    candidates = {
        "train": [
            feature_root / "train_manifest_corrected.csv",
            feature_root / "train_manifest.csv",
        ],
        "test": [
            feature_root / "test_manifest_corrected.csv",
            feature_root / "test_manifest.csv",
        ],
    }
    manifests: dict[str, pd.DataFrame] = {}
    for split, paths in candidates.items():
        for path in paths:
            if path.exists():
                frame = pd.read_csv(path)
                frame = frame.copy()
                frame["split"] = split
                manifests[split] = frame
                break
        else:
            raise FileNotFoundError(f"missing split manifest for {split}: {paths[0]}")
    return manifests


def build_channel_manifest(manifest: pd.DataFrame, channel: int) -> pd.DataFrame:
    frame = manifest.copy()
    if channel == 3:
        frame["file_path"] = frame["file_path"].astype(str).map(lambda value: str(_resolve_channel_source_path(Path(value), 3)))
        frame["channel"] = 3
    elif channel != 4:
        raise ValueError(f"unsupported channel: {channel}")
    frame["source_file_base"] = frame["file_path"].map(source_file_base_from_path)
    frame["split"] = frame["split"].astype(str)
    return frame.reset_index(drop=True)


def build_channel_manifests(feature_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_manifests = load_split_manifests(feature_root)
    ch3 = pd.concat([build_channel_manifest(split_manifests["train"], 3), build_channel_manifest(split_manifests["test"], 3)], ignore_index=True)
    ch4 = pd.concat([build_channel_manifest(split_manifests["train"], 4), build_channel_manifest(split_manifests["test"], 4)], ignore_index=True)
    return ch3, ch4


def verify_channel3_sources(feature_root: Path) -> dict[str, int]:
    split_manifests = load_split_manifests(feature_root)
    combined = pd.concat([split_manifests["train"], split_manifests["test"]], ignore_index=True)
    missing = 0
    for file_path in combined["file_path"].dropna().astype(str).unique():
        ch3_path = _resolve_channel_source_path(Path(file_path), 3)
        if not ch3_path.exists():
            missing += 1
    return {"channel4_files": int(combined["file_path"].nunique()), "channel3_missing": int(missing)}


def extract_channel_features(
    manifest: pd.DataFrame,
    channel_prefix: str,
    config: PipelineConfig,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    feature_names = _channel_feature_names(channel_prefix)
    for file_index, (file_path, file_rows) in enumerate(manifest.groupby("file_path", sort=True), start=1):
        raw = pd.read_csv(file_path)
        for row in file_rows.itertuples(index=False):
            samples = raw[str(row.source_column)].to_numpy(dtype=np.float64)
            row_split = getattr(row, "split", "all")
            row_source_base = getattr(row, "source_file_base", source_file_base_from_path(file_path))
            row_source_file = getattr(row, "source_file", row_source_base)
            row_channel = getattr(row, "channel", int(channel_prefix[-1]))
            row_original_fs = getattr(row, "original_fs", config.original_fs)
            row_speed_percent = getattr(row, "speed_percent", np.nan)
            row_raw_fault = getattr(row, "raw_fault", "")
            row_operating_condition = getattr(row, "operating_condition", getattr(row, "condition_id", ""))
            if not np.isfinite(samples).all():
                quality_rows.append({"split": row_split, "run_id": row.run_id, "reason": "raw_nonfinite"})
                continue
            processed = preprocess_run(samples, config)
            for window_index, (start, end, window) in enumerate(iter_windows(processed, config.window_size, config.step_size)):
                base_features = extract_candidate_features(window, rpm=row.rpm, config=config)
                channel_features = _build_channel_feature_values(base_features, window, config, channel_prefix)
                feature_row = OrderedDict()
                feature_row.update(
                    {
                        "sample_id": f"{row.run_id}_window_{window_index}",
                        "split": row_split,
                        "label": row.label,
                        "run_id": row.run_id,
                        "source_file_base": row_source_base,
                        "source_file": row_source_file,
                        "file_path": row.file_path,
                        "machine_id": row.machine_id,
                        "condition_id": row.condition_id,
                        "raw_fault": row_raw_fault,
                        "operating_condition": row_operating_condition,
                        "rpm": row.rpm,
                        "speed_percent": row_speed_percent,
                        "channel": row_channel,
                        "window_index": window_index,
                        "window_start": start / config.processed_fs,
                        "window_end": end / config.processed_fs,
                        "original_fs": row_original_fs,
                        "processed_fs": config.processed_fs,
                    }
                )
                feature_row.update({f"{channel_prefix}_{name}": float(channel_features[name]) for name in feature_names})
                rows.append(feature_row)
        if file_index % 50 == 0:
            print(f"{channel_prefix} feature extraction: {file_index}/{manifest['file_path'].nunique()}")

    frame = pd.DataFrame(rows)
    ordered_columns = [
        "sample_id",
        "split",
        "label",
        "run_id",
        "source_file_base",
        "source_file",
        "file_path",
        "machine_id",
        "condition_id",
        "raw_fault",
        "operating_condition",
        "rpm",
        "speed_percent",
        "channel",
        "window_index",
        "window_start",
        "window_end",
        "original_fs",
        "processed_fs",
    ] + [f"{channel_prefix}_{name}" for name in feature_names]
    frame = frame.loc[:, ordered_columns]
    frame.to_csv(output_path, index=False)
    quality_path = output_path.with_name(output_path.name.replace("features_raw.csv", "quality_report.csv"))
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
    return frame


def align_channel_windows(
    ch3: pd.DataFrame,
    ch4: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_columns = [column for column in ["split", "run_id", "source_file_base", "window_start", "window_end", "label"] if column in ch3.columns and column in ch4.columns]
    if "split" not in key_columns:
        key_columns = [column for column in ["run_id", "source_file_base", "window_start", "window_end", "label"] if column in ch3.columns and column in ch4.columns]
    ch3 = ch3.copy().reset_index(drop=True)
    ch4 = ch4.copy().reset_index(drop=True)

    ch3_indexed = ch3.set_index(key_columns)
    ch4_indexed = ch4.set_index(key_columns)
    common_index = ch3_indexed.index.intersection(ch4_indexed.index)
    aligned_ch3 = ch3_indexed.loc[common_index].reset_index()
    aligned_ch4 = ch4_indexed.loc[common_index].reset_index()
    aligned_ch3 = aligned_ch3.sort_values(key_columns).reset_index(drop=True)
    aligned_ch4 = aligned_ch4.sort_values(key_columns).reset_index(drop=True)
    fused = _build_fused_frame(aligned_ch3, aligned_ch4, key_columns)

    ch3_only = (
        ch3_indexed.loc[ch3_indexed.index.difference(common_index)]
        .reset_index()
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    ch4_only = (
        ch4_indexed.loc[ch4_indexed.index.difference(common_index)]
        .reset_index()
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    unmatched = pd.concat(
        [
            _annotate_unmatched(ch3_only, "channel3_only", "channel3"),
            _annotate_unmatched(ch4_only, "channel4_only", "channel4"),
        ],
        ignore_index=True,
    )
    return aligned_ch3, aligned_ch4, fused, unmatched


def build_balanced_training_subset(
    frame: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in BALANCE_MODES:
        raise ValueError(f"unsupported balance mode: {mode}")
    enriched = _ensure_balance_metadata(frame)
    if mode == "unbalanced":
        selected = enriched.copy().reset_index(drop=True)
    else:
        class_counts = enriched["class_name"].value_counts().sort_index()
        min_count = int(class_counts.min()) if not class_counts.empty else 0
        target_per_class = min_count if mode == "balanced_min" else int(np.floor(min_count * 1.5))
        selected_parts = []
        for class_name, class_frame in enriched.groupby("class_name", sort=True):
            selected_parts.append(_select_evenly_spaced_balanced_rows(class_frame, target_per_class))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else enriched.iloc[0:0].copy()
    distribution = _build_class_distribution(selected, mode)
    return selected, distribution


def filter_frame_by_balanced_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if keys.empty:
        return frame.iloc[0:0].copy()
    selected = frame.merge(keys[BALANCE_KEY_COLUMNS].drop_duplicates(), on=BALANCE_KEY_COLUMNS, how="inner", sort=False)
    return selected.reset_index(drop=True)


def _ensure_balance_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().reset_index(drop=True)
    enriched["class_name"] = enriched["label"].astype(str)
    enriched["source_file"] = enriched.get("source_file", enriched["source_file_base"]).astype(str)
    enriched["motor_id"] = enriched.get("motor_id", enriched["machine_id"]).astype(str)
    if "speed_percent" not in enriched.columns:
        enriched["speed_percent"] = np.nan
    enriched["speed_percent"] = enriched["speed_percent"].astype(float)
    enriched["speed_rpm"] = enriched.get("speed_rpm", enriched["rpm"]).astype(float)
    enriched["operating_condition"] = enriched.get("operating_condition", enriched["condition_id"]).astype(str)
    return enriched


def _build_class_distribution(frame: pd.DataFrame, balance_mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = _ensure_balance_metadata(frame).groupby("class_name", sort=True)
    for class_name, class_frame in grouped:
        rows.append(
            {
                "balance_mode": balance_mode,
                "class_name": class_name,
                "window_count": int(len(class_frame)),
                "source_file_count": int(class_frame["source_file"].nunique()),
                "motor_id_count": int(class_frame["motor_id"].nunique()),
                "speed_percent_count": int(class_frame["speed_percent"].nunique()),
                "speed_rpm_count": int(class_frame["speed_rpm"].nunique()),
                "operating_condition_count": int(class_frame["operating_condition"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _select_evenly_spaced_balanced_rows(class_frame: pd.DataFrame, target_count: int) -> pd.DataFrame:
    class_frame = _ensure_balance_metadata(class_frame)
    if target_count <= 0 or class_frame.empty:
        return class_frame.iloc[0:0].copy()
    if len(class_frame) <= target_count:
        return class_frame.sort_values(["source_file", "window_index", "window_start", "window_end"]).reset_index(drop=True)

    strata_sizes = class_frame.groupby(BALANCE_STRATA_COLUMNS, dropna=False, sort=True).size().reset_index(name="count")
    total = int(strata_sizes["count"].sum())
    raw_targets = strata_sizes["count"].to_numpy(dtype=np.float64) / max(total, 1) * target_count
    quotas = np.floor(raw_targets).astype(int)
    quotas = np.minimum(quotas, strata_sizes["count"].to_numpy(dtype=int))
    remaining = target_count - int(quotas.sum())
    fractional = raw_targets - np.floor(raw_targets)
    order = np.argsort(-fractional, kind="stable")
    while remaining > 0:
        progressed = False
        for idx in order:
            if quotas[idx] < int(strata_sizes.iloc[idx]["count"]):
                quotas[idx] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    selected_parts: list[pd.DataFrame] = []
    for quota, (_, strata_row) in zip(quotas, strata_sizes.iterrows()):
        if quota <= 0:
            continue
        mask = np.ones(len(class_frame), dtype=bool)
        for column in BALANCE_STRATA_COLUMNS:
            mask &= class_frame[column].astype(str).eq(str(strata_row[column])).to_numpy()
        strata_frame = class_frame.loc[mask].sort_values(["window_index", "window_start", "window_end"]).reset_index(drop=True)
        selected_parts.append(_sample_evenly_from_ordered_frame(strata_frame, int(quota)))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else class_frame.iloc[0:0].copy()
    selected = selected.sort_values(["source_file", "window_index", "window_start", "window_end"]).reset_index(drop=True)
    return selected


def _sample_evenly_from_ordered_frame(frame: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if target_count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if target_count >= len(frame):
        return frame.copy().reset_index(drop=True)
    positions = np.floor(np.linspace(0, len(frame), num=target_count, endpoint=False)).astype(int)
    positions = np.unique(np.clip(positions, 0, len(frame) - 1))
    if len(positions) < target_count:
        needed = target_count - len(positions)
        candidate_positions = [idx for idx in range(len(frame)) if idx not in set(positions)]
        positions = np.sort(np.concatenate([positions, np.asarray(candidate_positions[:needed], dtype=int)]))
    return frame.iloc[positions[:target_count]].reset_index(drop=True)


def build_alignment_reports(
    ch3: pd.DataFrame,
    ch4: pd.DataFrame,
    aligned_ch3: pd.DataFrame,
    aligned_ch4: pd.DataFrame,
    unmatched: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    group_columns = [column for column in ["split", "run_id", "source_file_base"] if column in ch3.columns and column in ch4.columns]
    report_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for keys, ch3_group in ch3.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        ch4_group = ch4
        for column, value in key_map.items():
            ch4_group = ch4_group[ch4_group[column] == value]
        aligned_group_ch3 = aligned_ch3
        aligned_group_ch4 = aligned_ch4
        for column, value in key_map.items():
            aligned_group_ch3 = aligned_group_ch3[aligned_group_ch3[column] == value]
            aligned_group_ch4 = aligned_group_ch4[aligned_group_ch4[column] == value]
        ch3_count = int(len(ch3_group))
        ch4_count = int(len(ch4_group))
        common_count = int(len(aligned_group_ch3))
        report_rows.append(
            {
                **key_map,
                "channel3_windows": ch3_count,
                "channel4_windows": ch4_count,
                "common_windows": common_count,
                "channel3_only_windows": int(ch3_count - common_count),
                "channel4_only_windows": int(ch4_count - common_count),
                "labels_match": bool(
                    set(zip(ch3_group["window_start"], ch3_group["window_end"], ch3_group["label"]))
                    == set(zip(ch4_group["window_start"], ch4_group["window_end"], ch4_group["label"]))
                ),
                "passed": ch3_count == ch4_count == common_count,
            }
        )
        if ch3_count != ch4_count or ch3_count != common_count:
            missing_rows.append({**key_map, "channel3_windows": ch3_count, "channel4_windows": ch4_count, "common_windows": common_count})

    alignment_report = pd.DataFrame(report_rows)
    missing_channels = pd.DataFrame(missing_rows)
    unmatched = unmatched.copy()
    alignment_report.to_csv(output_root / "channel34_alignment_report.csv", index=False)
    unmatched.to_csv(output_root / "channel34_unmatched_windows.csv", index=False)
    missing_channels.to_csv(output_root / "channel34_missing_channels.csv", index=False)
    return alignment_report, unmatched, missing_channels


def run_channel34_fusion_pipeline(
    feature_root: Path,
    config: PipelineConfig,
    modeling_config: ModelingConfig,
) -> dict[str, object]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    source_check = verify_channel3_sources(feature_root)
    if source_check["channel3_missing"] != 0:
        raise FileNotFoundError(f"missing channel 3 sources for {source_check['channel3_missing']} channel 4 files")
    split_manifests = load_split_manifests(feature_root)
    ch3_manifest = pd.concat([build_channel_manifest(split_manifests["train"], 3), build_channel_manifest(split_manifests["test"], 3)], ignore_index=True)
    ch4_manifest = pd.concat([build_channel_manifest(split_manifests["train"], 4), build_channel_manifest(split_manifests["test"], 4)], ignore_index=True)

    channel3_raw = extract_channel_features(ch3_manifest, "ch3", config, output_root / "channel3_features_raw.csv")
    channel4_raw = extract_channel_features(ch4_manifest, "ch4", config, output_root / "channel4_features_raw.csv")

    aligned_ch3, aligned_ch4, fused_raw, unmatched = align_channel_windows(channel3_raw, channel4_raw)
    aligned_ch3.to_csv(output_root / "channel3_aligned_features_raw.csv", index=False)
    aligned_ch4.to_csv(output_root / "channel4_aligned_features_raw.csv", index=False)
    fused_raw.to_csv(output_root / "channel34_fused_features_raw.csv", index=False)
    alignment_report, unmatched_windows, missing_channels = build_alignment_reports(
        channel3_raw,
        channel4_raw,
        aligned_ch3,
        aligned_ch4,
        unmatched,
        output_root,
    )

    scheme_frames = {
        "channel3": aligned_ch3,
        "channel4": aligned_ch4,
        "channel34_fused": fused_raw,
    }
    scheme_features = {
        "channel3": [column for column in aligned_ch3.columns if column.startswith("ch3_")],
        "channel4": [column for column in aligned_ch4.columns if column.startswith("ch4_")],
        "channel34_fused": [column for column in fused_raw.columns if column.startswith("ch3_") or column.startswith("ch4_")],
    }
    scheme_priority = {
        "channel3": CHANNEL3_PRIORITY_FEATURES,
        "channel4": CHANNEL4_PRIORITY_FEATURES,
        "channel34_fused": FUSION_PRIORITY_FEATURES,
    }

    train_reference = fused_raw[fused_raw["split"] == "train"].reset_index(drop=True)
    test_reference = fused_raw[fused_raw["split"] == "test"].reset_index(drop=True)
    class_distribution_before = _build_class_distribution(train_reference, "unbalanced")
    class_distribution_before.to_csv(output_root / "class_distribution_before.csv", index=False)

    modeling_root = output_root / "modeling"
    confusion_root = output_root / "channel34_confusion_matrices"
    modeling_root.mkdir(parents=True, exist_ok=True)
    confusion_root.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    cv_frames: list[pd.DataFrame] = []
    balancing_comparison_rows: list[dict[str, object]] = []
    selected_features_summary: dict[str, dict[str, list[str]]] = {}

    balanced_key_sets: dict[str, pd.DataFrame] = {}
    balanced_distribution_frames: dict[str, pd.DataFrame] = {}
    for balance_mode in BALANCE_MODES:
        balanced_reference, distribution = build_balanced_training_subset(train_reference, balance_mode)
        balanced_key_sets[balance_mode] = balanced_reference[BALANCE_KEY_COLUMNS].drop_duplicates().reset_index(drop=True)
        balanced_distribution_frames[balance_mode] = distribution
        if balance_mode == "balanced_min":
            distribution.to_csv(output_root / "class_distribution_after_min.csv", index=False)
        elif balance_mode == "balanced_1p5min":
            distribution.to_csv(output_root / "class_distribution_after_1p5min.csv", index=False)

        for _, row in distribution.iterrows():
            balancing_comparison_rows.append(
                {
                    "balance_mode": balance_mode,
                    "class_name": row["class_name"],
                    "window_count": int(row["window_count"]),
                    "source_file_count": int(row["source_file_count"]),
                    "motor_id_count": int(row["motor_id_count"]),
                    "speed_percent_count": int(row["speed_percent_count"]),
                    "speed_rpm_count": int(row["speed_rpm_count"]),
                    "operating_condition_count": int(row["operating_condition_count"]),
                }
            )

    for balance_mode, selected_train_keys in balanced_key_sets.items():
        for scheme_name, frame in scheme_frames.items():
            feature_columns = scheme_features[scheme_name]
            train = filter_frame_by_balanced_keys(frame[frame["split"] == "train"].reset_index(drop=True), selected_train_keys)
            test = frame[frame["split"] == "test"].reset_index(drop=True)
            selected_features_summary.setdefault(balance_mode, {})[scheme_name] = list(feature_columns)

            kept_valid_features, train_valid, test_valid, removed_invalid = remove_invalid_features(
                train[feature_columns],
                test[feature_columns],
                feature_columns,
            )
            selected_features, high_corr_pairs, removed_corr, _ = select_correlated_features(
                train_valid,
                kept_valid_features,
                threshold=0.95,
                protected_features=scheme_priority[scheme_name],
            )

            scheme_dir = modeling_root / scheme_name / balance_mode
            scheme_dir.mkdir(parents=True, exist_ok=True)
            removed_invalid_frame = removed_invalid.reindex(columns=["feature", "reason"])
            removed_invalid_frame.to_csv(scheme_dir / "removed_invalid_features.csv", index=False)
            high_corr_pairs.to_csv(scheme_dir / "high_correlation_pairs.csv", index=False)
            removed_corr.to_csv(scheme_dir / "removed_correlated_features.csv", index=False)
            pd.Series(selected_features).to_json(
                scheme_dir / "selected_features.json",
                force_ascii=False,
                indent=2,
            )

            metadata_columns = [column for column in train.columns if column not in feature_columns]
            train_selected = pd.concat([train[metadata_columns], train[selected_features]], axis=1)
            test_selected = pd.concat([test[[column for column in test.columns if column not in feature_columns]], test[selected_features]], axis=1)
            train_selected.to_csv(scheme_dir / "train_selected.csv", index=False)
            test_selected.to_csv(scheme_dir / "test_selected.csv", index=False)
            scaler, train_scaled, test_scaled = fit_scaler_and_transform(train_selected, test_selected, selected_features, scheme_dir)
            train_scaled.to_csv(scheme_dir / "train_scaled.csv", index=False)
            test_scaled.to_csv(scheme_dir / "test_scaled.csv", index=False)

            svm_result = tune_and_evaluate_model(
                "SVM",
                train_selected,
                test_selected,
                selected_features,
                modeling_config,
                groups=train_selected["file_path"],
                output_root=scheme_dir / "svm",
                scale=True,
            )
            mlp_result = tune_and_evaluate_model(
                "MLP",
                train_selected,
                test_selected,
                selected_features,
                modeling_config,
                groups=train_selected["file_path"],
                output_root=scheme_dir / "mlp",
                scale=True,
            )
            rf_result = tune_and_evaluate_model(
                "RandomForest",
                train_selected,
                test_selected,
                selected_features,
                modeling_config,
                groups=train_selected["file_path"],
                output_root=scheme_dir / "random_forest",
                scale=False,
            )

            results = [svm_result, mlp_result, rf_result]
            for result in results:
                cv_frame = result.cv_results.copy()
                if "model" in cv_frame.columns:
                    cv_frame = cv_frame.drop(columns=["model"])
                cv_frame.insert(0, "balance_mode", balance_mode)
                cv_frame.insert(1, "scheme", scheme_name)
                cv_frame.insert(2, "model", result.model_name)
                cv_frame.insert(3, "feature_count", len(selected_features))
                cv_frames.append(cv_frame)

                confusion_matrix_values = np.asarray(result.test_metrics["confusion_matrix"], dtype=int)
                error_counts = _channel34_error_counts(confusion_matrix_values, modeling_config.label_order)

                row = {
                    "balance_mode": balance_mode,
                    "scheme": scheme_name,
                    "model": result.model_name,
                    "feature_count": len(selected_features),
                    "selected_after_filter": len(selected_features),
                    "cv_macro_f1_mean": float(result.cv_results["macro_f1"].mean()) if not result.cv_results.empty else np.nan,
                    "cv_macro_f1_std": float(result.cv_results["macro_f1"].std(ddof=0)) if not result.cv_results.empty else np.nan,
                    "best_params": json.dumps(result.best_params, ensure_ascii=False),
                }
                row.update(
                    {
                        key: value
                        for key, value in result.test_metrics.items()
                        if key
                        not in {"classification_report", "confusion_matrix", "best_params", "model", "cv_macro_f1", "feature_count"}
                    }
                )
                row.update(error_counts)
                comparison_rows.append(row)

                summary = result.test_metrics["classification_report"]
                for label in modeling_config.label_order:
                    label_summary = summary.get(label, {})
                    per_class_rows.append(
                        {
                            "balance_mode": balance_mode,
                            "scheme": scheme_name,
                            "model": result.model_name,
                            "label": label,
                            "precision": float(label_summary.get("precision", 0.0)),
                            "recall": float(label_summary.get("recall", 0.0)),
                            "f1": float(label_summary.get("f1-score", 0.0)),
                            "support": int(label_summary.get("support", 0)),
                        }
                    )

                if result.model_name == "RandomForest":
                    importances = getattr(result.model, "feature_importances_", None)
                    if importances is not None:
                        ranked = sorted(zip(selected_features, importances), key=lambda item: item[1], reverse=True)
                        for rank, (feature, importance) in enumerate(ranked, start=1):
                            importance_rows.append(
                                {
                                    "balance_mode": balance_mode,
                                    "scheme": scheme_name,
                                    "model": result.model_name,
                                    "rank": rank,
                                    "feature": feature,
                                    "importance": float(importance),
                                }
                            )

                confusion_src = scheme_dir / result.model_name.lower() / f"{result.model_name}_confusion_matrix.csv"
                confusion_png = scheme_dir / result.model_name.lower() / f"{result.model_name}_confusion_matrix.png"
                confusion_dst_csv = confusion_root / f"{scheme_name}_{balance_mode}_{result.model_name}_confusion_matrix.csv"
                confusion_dst_png = confusion_root / f"{scheme_name}_{balance_mode}_{result.model_name}_confusion_matrix.png"
                if confusion_src.exists():
                    shutil.copy2(confusion_src, confusion_dst_csv)
                if confusion_png.exists():
                    shutil.copy2(confusion_png, confusion_dst_png)

    cv_comparison = pd.concat(cv_frames, ignore_index=True)
    cv_comparison.to_csv(output_root / "channel34_cv_comparison.csv", index=False)
    test_comparison = pd.DataFrame(comparison_rows)
    test_comparison.to_csv(output_root / "channel34_test_comparison.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(output_root / "channel34_per_class_metrics.csv", index=False)
    balancing_comparison = pd.DataFrame(balancing_comparison_rows)
    balancing_comparison.to_csv(output_root / "channel34_balancing_comparison.csv", index=False)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(output_root / "channel34_feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["balance_mode", "scheme", "model", "rank", "feature", "importance"]).to_csv(
            output_root / "channel34_feature_importance.csv",
            index=False,
        )
    pd.Series(selected_features_summary).to_json(output_root / "channel34_selected_features.json", force_ascii=False, indent=2)
    pd.DataFrame(
        [
            {
                "artifact": "channel3_raw",
                "rows": len(channel3_raw),
            },
            {
                "artifact": "channel4_raw",
                "rows": len(channel4_raw),
            },
            {
                "artifact": "channel34_fused",
                "rows": len(fused_raw),
            },
        ]
    ).to_csv(output_root / "channel34_artifact_summary.csv", index=False)

    summary = {
        "output_root": str(output_root),
        "source_check": source_check,
        "alignment_report_rows": int(len(alignment_report)),
        "unmatched_rows": int(len(unmatched_windows)),
        "missing_channels_rows": int(len(missing_channels)),
        "cv_rows": int(len(cv_comparison)),
        "test_rows": int(len(test_comparison)),
        "per_class_rows": int(len(per_class)),
        "balancing_rows": int(len(balancing_comparison)),
        "selected_features": selected_features_summary,
    }
    (output_root / "run_complete.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _channel_feature_names(channel_prefix: str) -> list[str]:
    return CHANNEL3_FEATURE_NAMES if channel_prefix == "ch3" else CHANNEL4_FEATURE_NAMES


def _build_channel_feature_values(
    base_features: OrderedDict[str, float],
    window: np.ndarray,
    config: PipelineConfig,
    channel_prefix: str,
) -> dict[str, float]:
    freqs, amps = amplitude_spectrum(window, config.processed_fs)
    power = np.square(amps)
    power_sum = float(np.sum(power)) + 1e-12
    wp_high = float(
        base_features["wp_energy_daa"]
        + base_features["wp_energy_dad"]
        + base_features["wp_energy_dda"]
        + base_features["wp_energy_ddd"]
    )
    channel_values = {
        "rot_1x_amp": float(base_features["amp_1x"]),
        "rot_2x_amp": float(base_features["amp_2x"]),
        "rot_3x_amp": float(base_features["amp_3x"]),
        "energy_1x": float(base_features["energy_1x"]),
        "energy_2x": float(base_features["energy_2x"]),
        "energy_3x": float(base_features["energy_3x"]),
        "amp_2x_div_1x": float(base_features["ratio_2x_to_1x"]),
        "amp_3x_div_1x": float(base_features["ratio_3x_to_1x"]),
        "harmonic_energy_1x_3x": float(base_features["rotation_band_energy"]),
        "rms": float(base_features["rms"]),
        "std": float(base_features["std"]),
        "peak_to_peak": float(base_features["peak_to_peak"]),
        "crest_factor": float(base_features["crest_factor"]),
        "impulse_factor": float(base_features["impulse_factor"]),
        "margin_factor": float(base_features["margin_factor"]),
        "clearance_factor": float(base_features["clearance_factor"]),
        "kurtosis": float(base_features["kurtosis"]),
        "spectral_centroid": float(base_features["freq_centroid"]),
        "spectral_entropy": float(base_features["spectral_entropy"]),
        "dominant_frequency": float(base_features["peak_frequency"]),
        "low_frequency_energy_ratio": float(base_features["band_ratio_10_500"]),
        "mid_frequency_energy_ratio": float(base_features["band_ratio_500_1000"]),
        "high_frequency_energy_ratio": float(base_features["band_ratio_2000_5000"]),
        "2000_5000_energy_ratio": float(base_features["band_ratio_2000_5000"]),
        "wavelet_packet_energy_entropy": float(base_features["wp_total_entropy"]),
        "wavelet_packet_high_frequency_energy": wp_high,
        "wp_high_frequency_energy_ratio": wp_high,
        "envelope_rms": float(base_features["env_rms"]),
        "envelope_kurtosis": float(base_features["env_kurtosis"]),
        "envelope_spectral_peak": float(base_features["env_peak_frequency"]),
        "envelope_spectral_entropy": float(base_features["env_spectral_entropy"]),
        "envelope_high_frequency_energy_ratio": float(base_features["env_ratio_1000_3000"]),
        "peak_interval_cv": _peak_interval_cv(window),
        "autocorrelation_peak": _autocorrelation_peak(window),
    }
    if channel_prefix == "ch3":
        channel_values["peak_frequency"] = float(base_features["peak_frequency"])
    return channel_values


def _build_fused_frame(aligned_ch3: pd.DataFrame, aligned_ch4: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    ch3_cols = [column for column in aligned_ch3.columns if column not in key_columns]
    ch4_cols = [column for column in aligned_ch4.columns if column not in key_columns]
    fused = aligned_ch3[key_columns].copy()
    fused = pd.concat([fused, aligned_ch3[ch3_cols].reset_index(drop=True), aligned_ch4[ch4_cols].reset_index(drop=True)], axis=1)
    fused = fused.loc[:, ~fused.columns.duplicated()]
    return fused


def _annotate_unmatched(frame: pd.DataFrame, reason: str, channel: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["channel", "reason", "split", "run_id", "source_file_base", "window_start", "window_end", "label"])
    annotated = frame.copy()
    annotated.insert(0, "channel", channel)
    annotated.insert(1, "reason", reason)
    return annotated


def _peak_interval_cv(window: np.ndarray) -> float:
    from scipy.signal import find_peaks

    abs_window = np.abs(np.asarray(window, dtype=np.float64))
    peaks, _ = find_peaks(abs_window)
    if peaks.size < 3:
        return 0.0
    intervals = np.diff(peaks).astype(np.float64)
    mean = float(np.mean(intervals))
    if mean == 0.0:
        return 0.0
    return float(np.std(intervals) / (mean + 1e-12))


def _autocorrelation_peak(window: np.ndarray) -> float:
    samples = np.asarray(window, dtype=np.float64)
    samples = samples - np.mean(samples)
    if not np.isfinite(samples).all() or samples.size < 2:
        return 0.0
    corr = np.correlate(samples, samples, mode="full")[samples.size - 1 :]
    if corr[0] == 0:
        return 0.0
    norm = corr / (corr[0] + 1e-12)
    if norm.size < 2:
        return 0.0
    return float(np.max(norm[1:]))


def _channel34_error_counts(matrix: np.ndarray, labels: tuple[str, ...]) -> dict[str, int]:
    index = {label: idx for idx, label in enumerate(labels)}
    pairs = {
        "misalignment_to_normal": ("联轴器不对中", "正常"),
        "bearing_to_looseness": ("轴承故障", "松动"),
        "normal_to_looseness": ("正常", "松动"),
    }
    return {
        key: int(matrix[index[left], index[right]]) if left in index and right in index else 0
        for key, (left, right) in pairs.items()
    }


def _resolve_channel_source_path(file_path: Path, channel: int) -> Path:
    name = file_path.name
    if channel == 4:
        if "-通道4.csv" in name:
            return file_path
        return file_path.with_name(re.sub(r"-通道\d+\.csv$", "-通道4.csv", name))
    if channel == 3:
        if "-通道3.csv" in name:
            return file_path
        return file_path.with_name(re.sub(r"-通道\d+\.csv$", "-通道3.csv", name))
    raise ValueError(f"unsupported channel: {channel}")
