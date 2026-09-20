"""Deterministic audit helpers for high-correlation feature pairs."""

from __future__ import annotations

import math

import pandas as pd

from .config import FEATURE_43


_SOURCE_COLUMNS = ["feature_a", "feature_b", "pearson_r", "channel", "split_mode"]
_SUMMARY_COLUMNS = [
    "feature_a",
    "feature_b",
    "occurrence_count",
    "channels",
    "split_modes",
    "min_abs_r",
    "max_abs_r",
    "signs",
]
_CHANNELS = {3, 4, 5}
_SPLIT_MODES = {"record", "temporal"}
_FEATURE_POSITION = {feature: position for position, feature in enumerate(FEATURE_43)}


def _canonical_pair(feature_a: str, feature_b: str) -> tuple[str, str]:
    if feature_a == feature_b:
        raise ValueError(f"self-pair is not allowed: {feature_a}")
    unknown = [feature for feature in (feature_a, feature_b) if feature not in _FEATURE_POSITION]
    if unknown:
        raise ValueError(f"unknown feature: {unknown[0]}")
    return tuple(sorted((feature_a, feature_b), key=_FEATURE_POSITION.__getitem__))


def consolidate_pairs(pairs: pd.DataFrame, include_source_rows: bool = False) -> pd.DataFrame:
    """Consolidate repeated unordered high-correlation pairs across baseline runs."""

    missing = [column for column in _SOURCE_COLUMNS if column not in pairs.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    output_columns = _SUMMARY_COLUMNS + (["source_rows"] if include_source_rows else [])
    if pairs.empty:
        return pd.DataFrame(columns=output_columns)

    normalized = []
    identities = set()
    for source in pairs[_SOURCE_COLUMNS].itertuples(index=False):
        feature_a, feature_b = _canonical_pair(source.feature_a, source.feature_b)
        try:
            pearson_r = float(source.pearson_r)
        except (TypeError, ValueError) as exc:
            raise ValueError("non-finite pearson_r") from exc
        if not math.isfinite(pearson_r):
            raise ValueError("non-finite pearson_r")
        if abs(pearson_r) < 0.95:
            raise ValueError(f"below correlation threshold: {pearson_r}")
        if source.channel not in _CHANNELS:
            raise ValueError(f"invalid channel: {source.channel}")
        if source.split_mode not in _SPLIT_MODES:
            raise ValueError(f"invalid split_mode: {source.split_mode}")

        identity = (source.channel, source.split_mode, feature_a, feature_b)
        if identity in identities:
            raise ValueError(f"duplicate source identity: {identity}")
        identities.add(identity)
        normalized.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "pearson_r": pearson_r,
                "channel": source.channel,
                "split_mode": source.split_mode,
            }
        )

    normalized.sort(
        key=lambda row: (
            _FEATURE_POSITION[row["feature_a"]],
            _FEATURE_POSITION[row["feature_b"]],
            row["channel"],
            row["split_mode"],
            row["pearson_r"],
        )
    )
    grouped = {}
    for row in normalized:
        grouped.setdefault((row["feature_a"], row["feature_b"]), []).append(row)

    summaries = []
    for (feature_a, feature_b), source_rows in grouped.items():
        absolute = [abs(row["pearson_r"]) for row in source_rows]
        summary = {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "occurrence_count": len(source_rows),
            "channels": sorted({row["channel"] for row in source_rows}),
            "split_modes": sorted({row["split_mode"] for row in source_rows}),
            "min_abs_r": min(absolute),
            "max_abs_r": max(absolute),
            "signs": sorted({"positive" if row["pearson_r"] > 0 else "negative" for row in source_rows}),
        }
        if include_source_rows:
            summary["source_rows"] = [
                {
                    "channel": row["channel"],
                    "split_mode": row["split_mode"],
                    "pearson_r": row["pearson_r"],
                }
                for row in source_rows
            ]
        summaries.append(summary)
    return pd.DataFrame(summaries, columns=output_columns)


_REMOVAL_REASONS = {
    "std": "near-duplicate of RMS for these vibration windows",
    "band_energy_3000_5000_ratio": "reference component of four-part closed composition",
    "wp_energy_ratio_7": "reference component of eight-part closed composition",
}
_LOCALLY_CORRELATED = {
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "spectral_centroid",
    "high_low_energy_ratio",
    "env_kurtosis",
    "env_crest_factor",
    "env_spectral_entropy",
    "env_peak_energy_ratio",
    "env_peak_concentration",
    "env_peak_count",
}


def _category(feature: str) -> str:
    if feature.startswith("env_"):
        return "envelope"
    if feature.startswith("wp_"):
        return "wavelet_packet"
    if feature.startswith(("rot_", "harmonic_", "noninteger_")):
        return "rotational_order"
    if feature.startswith("band_energy_") or feature in {
        "spectral_entropy",
        "spectral_flatness",
        "spectral_centroid",
        "spectral_bandwidth",
        "high_low_energy_ratio",
    }:
        return "spectral"
    return "time_domain"


def redundancy_decisions(features) -> pd.DataFrame:
    """Return the fixed, reviewable 43-to-40 feature redundancy decisions."""

    if list(features) != FEATURE_43:
        raise ValueError("features must be exactly FEATURE_43 in contract order")

    rows = []
    for feature in FEATURE_43:
        removed = feature in _REMOVAL_REASONS
        if removed:
            reason = _REMOVAL_REASONS[feature]
            review_note = "fixed ablation removal; retain the remaining component basis"
        elif feature in _LOCALLY_CORRELATED:
            reason = "retained as a physically distinct descriptor"
            review_note = "retained because correlation not stable across all channels"
        else:
            reason = "retained in the fixed 40-feature complement"
            review_note = "no globally stable redundancy removal selected"
        rows.append(
            {
                "feature": feature,
                "category": _category(feature),
                "decision": "remove" if removed else "retain",
                "reason": reason,
                "review_note": review_note,
            }
        )
    return pd.DataFrame(rows, columns=["feature", "category", "decision", "reason", "review_note"])
