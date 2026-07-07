from __future__ import annotations

from pathlib import Path

import pytest

from pump_diagnosis.inference_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import (
    FeatureVector,
    SampleMetadata,
    WindowPredictionResult,
)
from pump_fault_app.fusion import fuse_window_predictions


def _feature_vector(window_index: int) -> FeatureVector:
    return FeatureVector(
        feature_names=FORMAL_FEATURE_NAMES,
        values=tuple(float(window_index + feature_index) for feature_index in range(len(FORMAL_FEATURE_NAMES))),
        metadata=SampleMetadata(label=None, rpm=1480.0, device_id="Motor-2", channel=4),
    )


def _window_prediction(
    window_index: int,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
) -> WindowPredictionResult:
    return WindowPredictionResult(
        predicted_label=predicted_label,
        confidence=confidence,
        label_probabilities=probabilities,
        feature_vector=_feature_vector(window_index),
    )


def test_fuse_window_predictions_averages_probabilities_in_formal_label_order() -> None:
    first = _window_prediction(
        window_index=0,
        predicted_label="正常",
        confidence=0.60,
        probabilities={
            "正常": 0.60,
            "转子不平衡": 0.10,
            "联轴器不对中": 0.10,
            "松动": 0.05,
            "轴承故障": 0.10,
            "汽蚀": 0.05,
        },
    )
    second = _window_prediction(
        window_index=1,
        predicted_label="转子不平衡",
        confidence=0.50,
        probabilities={
            "正常": 0.20,
            "转子不平衡": 0.50,
            "联轴器不对中": 0.10,
            "松动": 0.10,
            "轴承故障": 0.05,
            "汽蚀": 0.05,
        },
    )

    result = fuse_window_predictions(
        (first, second),
        source_file=Path("demo.csv"),
    )

    assert tuple(result.label_probabilities.keys()) == FORMAL_LABEL_ORDER
    assert result.label_probabilities == {
        "正常": 0.40,
        "转子不平衡": 0.30,
        "联轴器不对中": 0.10,
        "松动": 0.075,
        "轴承故障": 0.075,
        "汽蚀": 0.05,
    }


def test_fuse_window_predictions_returns_top_label_confidence_and_window_count() -> None:
    first = _window_prediction(
        window_index=0,
        predicted_label="汽蚀",
        confidence=0.45,
        probabilities={
            "正常": 0.10,
            "转子不平衡": 0.10,
            "联轴器不对中": 0.10,
            "松动": 0.05,
            "轴承故障": 0.20,
            "汽蚀": 0.45,
        },
    )
    second = _window_prediction(
        window_index=1,
        predicted_label="轴承故障",
        confidence=0.50,
        probabilities={
            "正常": 0.10,
            "转子不平衡": 0.05,
            "联轴器不对中": 0.10,
            "松动": 0.05,
            "轴承故障": 0.50,
            "汽蚀": 0.20,
        },
    )

    result = fuse_window_predictions(
        [first, second],
        source_file=Path("record-01.csv"),
    )

    assert result.predicted_label == "轴承故障"
    assert result.confidence == 0.35
    assert result.window_count == 2
    assert result.source_file == Path("record-01.csv")


def test_fuse_window_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one window prediction is required"):
        fuse_window_predictions([], source_file=Path("empty.csv"))


def test_fuse_window_predictions_rejects_non_formal_probability_keys() -> None:
    invalid = _window_prediction(
        window_index=0,
        predicted_label="正常",
        confidence=0.90,
        probabilities={
            "正常": 0.90,
            "转子不平衡": 0.05,
            "联轴器不对中": 0.01,
            "松动": 0.01,
            "轴承故障": 0.01,
            "错误标签": 0.02,
        },
    )

    with pytest.raises(ValueError, match="window probability labels do not match formal label order"):
        fuse_window_predictions([invalid], source_file=Path("invalid.csv"))
