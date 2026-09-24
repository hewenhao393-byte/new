from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pump_fault_app.domain.diagnosis_models import (
    ChannelDiagnosisResult,
    ChannelName,
    WindowDiagnosisResult,
    probability_tuple,
)
from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER


def mean_probabilities(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FORMAL_LABEL_ORDER) or matrix.shape[0] == 0:
        raise ValueError("probability matrix must contain rows with six columns")
    return probability_tuple(matrix.mean(axis=0))


def fuse_window_probabilities(
    channel: ChannelName,
    windows: Sequence[WindowDiagnosisResult],
    *,
    quality_report=None,
    visualization=None,
    warnings: tuple[str, ...] = (),
) -> ChannelDiagnosisResult:
    if not windows:
        raise ValueError("at least one window prediction is required")
    probabilities = mean_probabilities([item.class_probabilities for item in windows])
    predicted_label = FORMAL_LABEL_ORDER[int(np.argmax(probabilities))]
    consistency = sum(item.predicted_label == predicted_label for item in windows) / len(windows)
    return ChannelDiagnosisResult(
        channel=channel,
        status="valid",
        predicted_label=predicted_label,
        class_probabilities=probabilities,
        window_predictions=tuple(windows),
        window_count=len(windows),
        valid_window_count=len(windows),
        window_consistency=float(consistency),
        quality_report=quality_report,
        visualization=visualization,
        warnings=warnings,
    )
