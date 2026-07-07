from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.inference_contract import FORMAL_LABEL_ORDER, validate_model_bundle
from pump_fault_app.domain.records import FeatureVector, WindowPredictionResult


@dataclass(frozen=True)
class FormalBPBundle:
    imputer: object
    scaler: object
    model: object
    feature_names: tuple[str, ...]
    formal_label_order: tuple[str, ...]


def load_formal_bp_bundle(bundle_path: Path | str | None = None) -> FormalBPBundle:
    bundle = validate_model_bundle(bundle_path)
    return FormalBPBundle(
        imputer=bundle["imputer"],
        scaler=bundle["scaler"],
        model=bundle["model"],
        feature_names=tuple(bundle["features"]),
        formal_label_order=FORMAL_LABEL_ORDER,
    )


def predict_single_window(feature_vector: FeatureVector, bundle: FormalBPBundle) -> WindowPredictionResult:
    if feature_vector.feature_names != bundle.feature_names:
        raise ValueError("feature vector names do not match formal bundle feature order")

    raw = pd.DataFrame([feature_vector.values], columns=bundle.feature_names, dtype=np.float64)
    imputed = bundle.imputer.transform(raw)
    scaled = bundle.scaler.transform(imputed)
    probabilities = np.asarray(bundle.model.predict_proba(scaled), dtype=np.float64)
    ordered = _reorder_probabilities(probabilities[0], tuple(str(label) for label in bundle.model.classes_), bundle.formal_label_order)

    best_label = max(ordered, key=ordered.get)
    return WindowPredictionResult(
        predicted_label=best_label,
        confidence=float(ordered[best_label]),
        label_probabilities=ordered,
        feature_vector=feature_vector,
    )


def _reorder_probabilities(
    probability_row: np.ndarray,
    model_classes: tuple[str, ...],
    formal_label_order: tuple[str, ...],
) -> dict[str, float]:
    class_to_probability = {
        str(label): float(probability_row[index])
        for index, label in enumerate(model_classes)
    }
    return {label: class_to_probability[label] for label in formal_label_order}
