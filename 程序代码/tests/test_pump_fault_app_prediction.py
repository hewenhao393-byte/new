from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import FeatureVector, SampleMetadata
from pump_fault_app.prediction.catboost_loader import DEFAULT_MODEL_DIRECTORY, load_catboost43_models
from pump_fault_app.prediction.channel_predictor import predict_channel_window


class _FakeModel:
    def __init__(self) -> None:
        self.classes_ = np.array(
            ["松动", "正常", "汽蚀", "联轴器不对中", "转子不平衡", "轴承故障"], dtype=object
        )

    def predict_proba(self, values):
        assert tuple(values.columns) == FORMAL_FEATURE_NAMES
        return np.array([[0.10, 0.20, 0.15, 0.05, 0.40, 0.10]], dtype=np.float64)


def _feature_vector(names: tuple[str, ...] = FORMAL_FEATURE_NAMES) -> FeatureVector:
    return FeatureVector(
        feature_names=names,
        values=tuple(float(index + 1) for index in range(len(names))),
        metadata=SampleMetadata(rpm=1480.0, channel=3),
    )


def test_loader_accepts_the_published_three_channel_models() -> None:
    loaded = load_catboost43_models(DEFAULT_MODEL_DIRECTORY)
    assert tuple(loaded.models) == ("CH3", "CH4", "CH5")
    assert loaded.feature_names == FORMAL_FEATURE_NAMES
    assert loaded.label_order == FORMAL_LABEL_ORDER


def test_loader_rejects_manifest_feature_reordering(tmp_path: Path) -> None:
    manifest = json.loads((DEFAULT_MODEL_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    manifest["features"] = list(reversed(FORMAL_FEATURE_NAMES))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        load_catboost43_models(tmp_path)


def test_loader_rejects_incorrect_model_hash(tmp_path: Path) -> None:
    manifest = json.loads((DEFAULT_MODEL_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    manifest["models"]["CH3"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    for channel in ("ch3", "ch4", "ch5"):
        (tmp_path / f"{channel}.cbm").symlink_to(DEFAULT_MODEL_DIRECTORY / f"{channel}.cbm")
    with pytest.raises(ValueError, match="SHA-256"):
        load_catboost43_models(tmp_path)


def test_predictor_reorders_model_classes_to_formal_order() -> None:
    result = predict_channel_window(_feature_vector(), _FakeModel(), "CH3")
    assert result.class_probabilities == pytest.approx((0.20, 0.40, 0.05, 0.10, 0.10, 0.15))
    assert result.predicted_label == "转子不平衡"
    assert (result.window_index, result.start_index, result.end_index) == (0, 0, 4800)


def test_predictor_rejects_reordered_feature_vector() -> None:
    with pytest.raises(ValueError, match="feature order"):
        predict_channel_window(_feature_vector(tuple(reversed(FORMAL_FEATURE_NAMES))), _FakeModel(), "CH3")
