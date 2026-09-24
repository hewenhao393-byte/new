from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pump_fault_app.domain.diagnosis_models import (
    ChannelDiagnosisResult,
    MultiChannelDiagnosisResult,
)
from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.fusion.window_fusion import mean_probabilities


def _agreement(labels: tuple[str, ...]) -> tuple[str | None, int]:
    most_common = Counter(labels).most_common(1)[0][1]
    if len(labels) == 1:
        return None, 1
    if most_common == len(labels):
        return "高一致", most_common
    if len(labels) == 3 and most_common == 2:
        return "中等一致", most_common
    return "低一致", most_common


def fuse_valid_channels(
    channel_results: Sequence[ChannelDiagnosisResult],
) -> MultiChannelDiagnosisResult:
    if not channel_results:
        raise ValueError("at least one channel result is required")
    ordered = tuple(channel_results)
    input_channels = tuple(item.channel for item in ordered)
    if len(set(input_channels)) != len(input_channels):
        raise ValueError("duplicate channel results")
    valid = tuple(item for item in ordered if item.status == "valid")
    invalid = tuple(item for item in ordered if item.status == "invalid")
    warnings = tuple(warning for item in ordered for warning in item.warnings)
    if not valid:
        return MultiChannelDiagnosisResult(
            status="failed",
            input_channels=input_channels,
            valid_channels=(),
            invalid_channels=tuple(item.channel for item in invalid),
            channel_results=ordered,
            predicted_label=None,
            fused_probabilities=None,
            agreement_level=None,
            agreement_count=0,
            valid_channel_count=0,
            warnings=warnings,
        )

    fused = mean_probabilities([item.class_probabilities for item in valid if item.class_probabilities])
    predicted_label = FORMAL_LABEL_ORDER[int(np.argmax(fused))]
    agreement_level, agreement_count = _agreement(
        tuple(item.predicted_label for item in valid if item.predicted_label is not None)
    )
    return MultiChannelDiagnosisResult(
        status="diagnosed",
        input_channels=input_channels,
        valid_channels=tuple(item.channel for item in valid),
        invalid_channels=tuple(item.channel for item in invalid),
        channel_results=ordered,
        predicted_label=predicted_label,
        fused_probabilities=fused,
        agreement_level=agreement_level,
        agreement_count=agreement_count,
        valid_channel_count=len(valid),
        warnings=warnings,
    )
