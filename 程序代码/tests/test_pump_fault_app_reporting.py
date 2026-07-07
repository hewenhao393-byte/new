from __future__ import annotations

from pathlib import Path

from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference
from pump_fault_app.reporting import build_diagnosis_summary
from tests.test_pump_fault_app_inference import (
    _write_fake_bundle,
    _write_signal_csv,
    _write_unknown_warning_bundle,
    _write_warning_bundle,
)

import numpy as np


def test_build_diagnosis_summary_returns_success_payload(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
        + 0.05 * np.random.default_rng(11).normal(size=4800)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            device_id="Motor-2",
            measurement_position="泵端",
        )
    )

    summary = build_diagnosis_summary(inference_result)

    assert summary.success is True
    assert summary.status == "diagnosed"
    assert summary.diagnosis_label == "转子不平衡"
    assert summary.confidence == 0.40
    assert summary.window_count == 3
    assert summary.message == "诊断完成"
    assert summary.warnings == ()
    payload = summary.as_dict()
    assert payload["file_name"] == "record.csv"
    assert payload["sampling_rate_hz"] == 12000
    assert payload["rpm"] == 1500.0
    assert payload["top_probabilities"][0]["label"] == "转子不平衡"


def test_build_diagnosis_summary_returns_quality_failure_payload(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "too_short.csv"
    _write_signal_csv(signal_path, np.ones(1200, dtype=np.float64))

    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    summary = build_diagnosis_summary(inference_result)

    assert summary.success is False
    assert summary.status == "rejected"
    assert summary.diagnosis_label is None
    assert summary.confidence is None
    assert summary.window_count is None
    assert summary.message == "信号质量不满足诊断条件"
    assert "signal length is shorter than one formal window" in summary.rejection_reasons


def test_build_diagnosis_summary_returns_input_failure_payload(tmp_path: Path) -> None:
    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=tmp_path / "missing.csv",
            sampling_rate_hz=12000,
            rpm=1500.0,
        )
    )

    summary = build_diagnosis_summary(inference_result)

    assert summary.success is False
    assert summary.status == "input_error"
    assert summary.message == "输入文件读取失败"
    assert summary.failure_stage == "input"
    assert "file does not exist" in str(summary.failure_message)


def test_build_diagnosis_summary_exposes_runtime_warnings(tmp_path: Path) -> None:
    bundle_path = tmp_path / "warning_bundle.joblib"
    _write_warning_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    summary = build_diagnosis_summary(inference_result)

    assert summary.runtime_warnings
    assert "overflow encountered in matmul" in summary.runtime_warnings[0]


def test_build_diagnosis_summary_maps_runtime_warnings_to_alerts(tmp_path: Path) -> None:
    bundle_path = tmp_path / "warning_bundle.joblib"
    _write_warning_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    summary = build_diagnosis_summary(
        run_formal_inference(
            FormalInferenceRequest(
                file_path=signal_path,
                sampling_rate_hz=12000,
                rpm=1500.0,
                model_bundle_path=bundle_path,
            )
        )
    )

    assert summary.runtime_alerts
    assert summary.runtime_alerts[0]["code"] == "numeric_stability_warning"
    assert summary.runtime_alerts[0]["severity"] == "warning"
    assert "模型推理过程中出现数值稳定性告警" in summary.runtime_alerts[0]["message"]


def test_build_diagnosis_summary_maps_unknown_runtime_warning_to_generic_alert(tmp_path: Path) -> None:
    bundle_path = tmp_path / "unknown_warning_bundle.joblib"
    _write_unknown_warning_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    summary = build_diagnosis_summary(
        run_formal_inference(
            FormalInferenceRequest(
                file_path=signal_path,
                sampling_rate_hz=12000,
                rpm=1500.0,
                model_bundle_path=bundle_path,
            )
        )
    )

    assert summary.runtime_alerts
    assert summary.runtime_alerts[0]["code"] == "runtime_warning"
    assert "模型推理过程中出现运行告警" in summary.runtime_alerts[0]["message"]
