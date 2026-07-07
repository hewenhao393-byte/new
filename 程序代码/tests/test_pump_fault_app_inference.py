from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import warnings

from pump_diagnosis.inference_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference


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


class _WarningModel(_FakeModel):
    def predict_proba(self, values):
        warnings.warn("overflow encountered in matmul", RuntimeWarning)
        return super().predict_proba(values)


class _UnknownWarningModel(_FakeModel):
    def predict_proba(self, values):
        warnings.warn("some new runtime warning", RuntimeWarning)
        return super().predict_proba(values)


def _write_signal_csv(path: Path, samples: np.ndarray) -> None:
    lines = ["time,channel1,通道4"]
    for index, value in enumerate(samples):
        lines.append(f"{index / 12000:.6f},0.0,{value:.10f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_fake_bundle(path: Path) -> None:
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _FakeModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        path,
    )


def _write_warning_bundle(path: Path) -> None:
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _WarningModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        path,
    )


def _write_unknown_warning_bundle(path: Path) -> None:
    joblib.dump(
        {
            "imputer": _FakeImputer(),
            "scaler": _FakeScaler(),
            "model": _UnknownWarningModel(),
            "features": list(FORMAL_FEATURE_NAMES),
        },
        path,
    )


def test_run_formal_inference_returns_record_level_prediction(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
        + 0.05 * np.random.default_rng(7).normal(size=4800)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    assert result.success is True
    assert result.failure_stage is None
    assert result.failure_message is None
    assert result.raw_signal is not None
    assert result.quality_report is not None
    assert result.preprocessed_signal is not None
    assert result.windowing_result is not None
    assert result.window_predictions is not None
    assert result.record_prediction is not None
    assert tuple(result.record_prediction.label_probabilities.keys()) == FORMAL_LABEL_ORDER
    assert result.record_prediction.predicted_label == "转子不平衡"
    assert result.record_prediction.confidence == 0.40
    assert result.record_prediction.window_count == 3


def test_run_formal_inference_returns_quality_failure_without_predictions(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "too_short.csv"
    _write_signal_csv(signal_path, np.ones(1200, dtype=np.float64))

    result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    assert result.success is False
    assert result.failure_stage == "quality"
    assert result.failure_message is not None
    assert result.quality_report is not None
    assert result.quality_report.allow_diagnosis is False
    assert result.preprocessed_signal is None
    assert result.windowing_result is None
    assert result.window_predictions is None
    assert result.record_prediction is None


def test_run_formal_inference_surfaces_read_failures_as_input_stage(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    result = run_formal_inference(
        FormalInferenceRequest(
            file_path=missing_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
        )
    )

    assert result.success is False
    assert result.failure_stage == "input"
    assert "file does not exist" in str(result.failure_message)
    assert result.raw_signal is None
    assert result.quality_report is None


def test_run_formal_inference_collects_runtime_warnings(tmp_path: Path) -> None:
    bundle_path = tmp_path / "warning_bundle.joblib"
    _write_warning_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    assert result.success is True
    assert result.runtime_warnings
    assert "overflow encountered in matmul" in result.runtime_warnings[0]
