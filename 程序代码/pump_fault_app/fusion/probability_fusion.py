from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import RecordPredictionResult, WindowPredictionResult


def fuse_window_predictions(
    window_predictions: Iterable[WindowPredictionResult],
    *,
    source_file: Path,
) -> RecordPredictionResult:
    predictions = tuple(window_predictions)
    if not predictions:
        raise ValueError("at least one window prediction is required")

    _validate_probability_labels(predictions)

    averaged_probabilities = {
        label: round(
            sum(prediction.label_probabilities[label] for prediction in predictions) / len(predictions),
            12,
        )
        for label in FORMAL_LABEL_ORDER
    }
    predicted_label = max(averaged_probabilities, key=averaged_probabilities.get)

    return RecordPredictionResult(
        source_file=source_file,
        predicted_label=predicted_label,
        confidence=float(averaged_probabilities[predicted_label]),
        label_probabilities=averaged_probabilities,
        window_count=len(predictions),
        window_predictions=predictions,
    )


def _validate_probability_labels(window_predictions: tuple[WindowPredictionResult, ...]) -> None:
    for prediction in window_predictions:
        if tuple(prediction.label_probabilities.keys()) != FORMAL_LABEL_ORDER:
            raise ValueError("window probability labels do not match formal label order")
