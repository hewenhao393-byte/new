from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


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


def select_representative_windows(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    missing_columns = [column for column in REDRAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"frame is missing required redraw feature columns: {missing_list}")

    selected: dict[str, dict[str, object]] = {}
    if frame.empty:
        return selected

    feature_frame = frame.loc[:, REDRAW_FEATURE_COLUMNS].astype(float)
    for label, label_frame in frame.groupby("label", sort=False):
        label_features = feature_frame.loc[label_frame.index]
        center = label_features.mean(axis=0).to_numpy(dtype=float)
        distances = np.linalg.norm(label_features.to_numpy(dtype=float) - center, axis=1)
        best_pos = int(np.argmin(distances))
        row = label_frame.iloc[best_pos]
        selected[label] = row.to_dict()
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
    return any(float(peak) > 300.0 for peak in peaks_hz)
