from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import CHANNELS, LABELS


class FusionError(ValueError):
    """Raised when probability fusion inputs violate the contract."""


def _normalize_scores(scores: Mapping[int, float]) -> dict[int, float]:
    values = {channel: float(scores[channel]) for channel in CHANNELS}
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise FusionError("weights must be finite and non-negative")
    total = sum(values.values())
    if total == 0:
        return {channel: 1.0 / len(CHANNELS) for channel in CHANNELS}
    return {channel: value / total for channel, value in values.items()}


def derive_global_weights(channel_macro_f1: Mapping[int, float]) -> dict[int, float]:
    if set(channel_macro_f1) != set(CHANNELS):
        raise FusionError("global scores must contain CH3, CH4, and CH5")
    return _normalize_scores(channel_macro_f1)


def derive_classwise_weights(
    channel_class_f1: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    if set(channel_class_f1) != set(CHANNELS):
        raise FusionError("classwise scores must contain CH3, CH4, and CH5")
    result = {channel: {} for channel in CHANNELS}
    for label in LABELS:
        normalized = _normalize_scores({channel: channel_class_f1[channel][label] for channel in CHANNELS})
        for channel in CHANNELS:
            result[channel][label] = normalized[channel]
    return result


def _validate_probabilities(probabilities: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    if not probabilities or not set(probabilities).issubset(CHANNELS):
        raise FusionError("one or more of CH3, CH4, CH5 probabilities are required")
    arrays = {channel: np.asarray(values, dtype=float) for channel, values in probabilities.items()}
    row_count = None
    for channel, values in arrays.items():
        if values.ndim != 2 or values.shape[1] != len(LABELS):
            raise FusionError(f"CH{channel} must be a two-dimensional six-class probability matrix")
        if row_count is None:
            row_count = values.shape[0]
        if values.shape[0] != row_count:
            raise FusionError("channel probability row counts differ")
        if not np.isfinite(values).all() or (values < 0).any():
            raise FusionError("probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
            raise FusionError("each channel probability row must sum to one")
    return arrays


def fuse_probabilities(
    probabilities: Mapping[int, np.ndarray],
    method: str,
    weights: Mapping | None = None,
) -> np.ndarray:
    arrays = _validate_probabilities(probabilities)
    available = tuple(sorted(arrays))
    if method == "equal":
        fused = np.mean([arrays[channel] for channel in available], axis=0)
    elif method == "global":
        if weights is None:
            raise FusionError("global weights are required")
        raw = np.asarray([float(weights[channel]) for channel in available])
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise FusionError("global weights must be finite and non-negative")
        normalized = np.full(len(available), 1 / len(available)) if raw.sum() == 0 else raw / raw.sum()
        fused = sum(normalized[i] * arrays[channel] for i, channel in enumerate(available))
    elif method == "classwise":
        if weights is None:
            raise FusionError("classwise weights are required")
        fused = np.zeros_like(next(iter(arrays.values())))
        for class_index, label in enumerate(LABELS):
            raw = np.asarray([float(weights[channel][label]) for channel in available])
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise FusionError("classwise weights must be finite and non-negative")
            normalized = np.full(len(available), 1 / len(available)) if raw.sum() == 0 else raw / raw.sum()
            fused[:, class_index] = sum(
                normalized[i] * arrays[channel][:, class_index]
                for i, channel in enumerate(available)
            )
        totals = fused.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise FusionError("classwise fused scores cannot be normalized")
        fused = fused / totals
    else:
        raise FusionError(f"unknown fusion method: {method}")
    return fused
