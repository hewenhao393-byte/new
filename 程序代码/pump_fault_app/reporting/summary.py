from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pump_fault_app.inference import FormalInferenceResult


@dataclass(frozen=True)
class DiagnosisSummary:
    success: bool
    status: str
    message: str
    file_name: str | None
    sampling_rate_hz: int | None
    rpm: float | None
    device_id: str | None
    measurement_position: str | None
    diagnosis_label: str | None
    confidence: float | None
    window_count: int | None
    failure_stage: str | None
    failure_message: str | None
    warnings: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    top_probabilities: tuple[dict[str, float | str], ...] = ()
    runtime_warnings: tuple[str, ...] = ()
    runtime_alerts: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_diagnosis_summary(result: FormalInferenceResult) -> DiagnosisSummary:
    raw_signal = result.raw_signal
    quality_report = result.quality_report
    record_prediction = result.record_prediction
    runtime_alerts = _build_runtime_alerts(result.runtime_warnings)

    if result.success and raw_signal is not None and quality_report is not None and record_prediction is not None:
        ranked_probabilities = tuple(
            {"label": label, "probability": probability}
            for label, probability in sorted(
                record_prediction.label_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return DiagnosisSummary(
            success=True,
            status="diagnosed",
            message="诊断完成",
            file_name=raw_signal.file_name,
            sampling_rate_hz=raw_signal.sampling_rate_hz,
            rpm=raw_signal.rpm,
            device_id=raw_signal.device_id,
            measurement_position=raw_signal.measurement_position,
            diagnosis_label=record_prediction.predicted_label,
            confidence=record_prediction.confidence,
            window_count=record_prediction.window_count,
            failure_stage=None,
            failure_message=None,
            warnings=quality_report.warnings,
            rejection_reasons=(),
            top_probabilities=ranked_probabilities,
            runtime_warnings=result.runtime_warnings,
            runtime_alerts=runtime_alerts,
        )

    if result.failure_stage == "quality" and raw_signal is not None and quality_report is not None:
        return DiagnosisSummary(
            success=False,
            status="rejected",
            message="信号质量不满足诊断条件",
            file_name=raw_signal.file_name,
            sampling_rate_hz=raw_signal.sampling_rate_hz,
            rpm=raw_signal.rpm,
            device_id=raw_signal.device_id,
            measurement_position=raw_signal.measurement_position,
            diagnosis_label=None,
            confidence=None,
            window_count=None,
            failure_stage=result.failure_stage,
            failure_message=result.failure_message,
            warnings=quality_report.warnings,
            rejection_reasons=quality_report.rejection_reasons,
            top_probabilities=(),
            runtime_warnings=result.runtime_warnings,
            runtime_alerts=runtime_alerts,
        )

    return DiagnosisSummary(
        success=False,
        status="input_error" if result.failure_stage == "input" else "failed",
        message="输入文件读取失败" if result.failure_stage == "input" else "诊断失败",
        file_name=raw_signal.file_name if raw_signal is not None else None,
        sampling_rate_hz=raw_signal.sampling_rate_hz if raw_signal is not None else None,
        rpm=raw_signal.rpm if raw_signal is not None else None,
        device_id=raw_signal.device_id if raw_signal is not None else None,
        measurement_position=raw_signal.measurement_position if raw_signal is not None else None,
        diagnosis_label=None,
        confidence=None,
        window_count=None,
        failure_stage=result.failure_stage,
        failure_message=result.failure_message,
        warnings=(),
        rejection_reasons=(),
        top_probabilities=(),
        runtime_warnings=result.runtime_warnings,
        runtime_alerts=runtime_alerts,
    )


def _build_runtime_alerts(runtime_warnings: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    alerts: list[dict[str, str]] = []
    for warning_text in runtime_warnings:
        normalized = warning_text.lower()
        if any(keyword in normalized for keyword in ("overflow", "underflow", "divide by zero", "invalid value", "matmul")):
            alerts.append(
                {
                    "code": "numeric_stability_warning",
                    "severity": "warning",
                    "message": "模型推理过程中出现数值稳定性告警，结果可用但建议复核输入信号与模型状态。",
                    "source_warning": warning_text,
                }
            )
            continue
        alerts.append(
            {
                "code": "runtime_warning",
                "severity": "warning",
                "message": "模型推理过程中出现运行告警，建议结合原始告警信息复核本次诊断结果。",
                "source_warning": warning_text,
            }
        )
    return tuple(alerts)
