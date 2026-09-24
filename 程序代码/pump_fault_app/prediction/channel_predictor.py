from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pump_fault_app.domain.diagnosis_models import ChannelName, WindowDiagnosisResult, probability_tuple
from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import FeatureVector


def predict_channel_window(
    feature_vector: FeatureVector,
    model: Any,
    channel: ChannelName,
    *,
    window_index: int = 0,
    start_index: int = 0,
    end_index: int = 4800,
) -> WindowDiagnosisResult:
    if channel not in {"CH3", "CH4", "CH5"}:
        raise ValueError(f"unsupported channel: {channel}")
    if feature_vector.feature_names != FORMAL_FEATURE_NAMES:
        raise ValueError("feature vector does not match the frozen feature order")
    values = np.asarray(feature_vector.values, dtype=np.float64)
    if values.shape != (len(FORMAL_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("feature vector must contain 43 finite values")

    frame = pd.DataFrame([values], columns=FORMAL_FEATURE_NAMES, dtype=np.float64)
    raw_probabilities = np.asarray(model.predict_proba(frame), dtype=np.float64)
    if raw_probabilities.shape != (1, len(FORMAL_LABEL_ORDER)):
        raise ValueError("model probability output must contain one row and six columns")
    model_classes = tuple(str(label) for label in model.classes_)
    if len(model_classes) != len(FORMAL_LABEL_ORDER) or set(model_classes) != set(FORMAL_LABEL_ORDER):
        raise ValueError("model classes do not match the formal six-label set")
    by_label = dict(zip(model_classes, raw_probabilities[0]))
    ordered = probability_tuple(by_label[label] for label in FORMAL_LABEL_ORDER)
    predicted_label = FORMAL_LABEL_ORDER[int(np.argmax(ordered))]
    return WindowDiagnosisResult(
        window_index=window_index,
        start_index=start_index,
        end_index=end_index,
        predicted_label=predicted_label,
        class_probabilities=ordered,
    )
