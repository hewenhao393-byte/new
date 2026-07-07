from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from pump_diagnosis.inference_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import FeatureVector, SampleMetadata
from pump_fault_app.prediction.predictor import (
    load_formal_bp_bundle,
    predict_single_window,
)


class _FakeImputer:
    def transform(self, values):
        return values


class _FakeScaler:
    def transform(self, values):
        return values


class _FakeModel:
    def __init__(self) -> None:
        self.classes_ = np.array(["松动", "正常", "汽蚀", "联轴器不对中", "转子不平衡", "轴承故障"], dtype=object)

    def predict_proba(self, values):
        _ = values
        return np.array([[0.10, 0.20, 0.15, 0.05, 0.40, 0.10]], dtype=np.float64)


def _feature_vector() -> FeatureVector:
    return FeatureVector(
        feature_names=FORMAL_FEATURE_NAMES,
        values=tuple(float(index + 1) for index in range(len(FORMAL_FEATURE_NAMES))),
        metadata=SampleMetadata(label=None, rpm=1480.0, device_id="Motor-2"),
    )


def test_load_formal_bp_bundle_returns_bundle_parts(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _FakeModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        bundle_path,
    )

    bundle = load_formal_bp_bundle(bundle_path)

    assert bundle.feature_names == FORMAL_FEATURE_NAMES
    assert tuple(bundle.formal_label_order) == FORMAL_LABEL_ORDER


def test_predict_single_window_reorders_probabilities_to_formal_label_order(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _FakeModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        bundle_path,
    )

    bundle = load_formal_bp_bundle(bundle_path)
    result = predict_single_window(_feature_vector(), bundle)

    assert result.label_probabilities == {
        "正常": 0.20,
        "转子不平衡": 0.40,
        "联轴器不对中": 0.05,
        "松动": 0.10,
        "轴承故障": 0.10,
        "汽蚀": 0.15,
    }


def test_predict_single_window_returns_formal_top_label_and_confidence(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _FakeModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        bundle_path,
    )

    bundle = load_formal_bp_bundle(bundle_path)
    result = predict_single_window(_feature_vector(), bundle)

    assert result.predicted_label == "转子不平衡"
    assert result.confidence == 0.40
    assert result.feature_vector.feature_names == FORMAL_FEATURE_NAMES
