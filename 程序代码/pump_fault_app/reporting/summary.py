from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult
from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.domain.labels import display_label


@dataclass(frozen=True)
class DiagnosisSummary:
    success: bool
    status: str
    message: str
    input_channels: tuple[str, ...]
    valid_channels: tuple[str, ...]
    invalid_channels: tuple[str, ...]
    diagnosis_label: str | None
    model_output_probabilities: tuple[dict[str, float | str], ...]
    agreement_level: str | None
    agreement_count: int
    model_version: str
    feature_version: str
    contract_version: str
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_diagnosis_summary(result: MultiChannelDiagnosisResult) -> DiagnosisSummary:
    probabilities = ()
    if result.fused_probabilities is not None:
        probabilities = tuple(
            {"label": display_label(label), "internal_label": label, "probability": probability}
            for label, probability in zip(FORMAL_LABEL_ORDER, result.fused_probabilities)
        )
    return DiagnosisSummary(
        success=result.status == "diagnosed",
        status=result.status,
        message="诊断完成" if result.status == "diagnosed" else "无有效通道，诊断失败",
        input_channels=result.input_channels,
        valid_channels=result.valid_channels,
        invalid_channels=result.invalid_channels,
        diagnosis_label=display_label(result.predicted_label) if result.predicted_label else None,
        model_output_probabilities=probabilities,
        agreement_level=result.agreement_level,
        agreement_count=result.agreement_count,
        model_version=result.model_version,
        feature_version=result.feature_version,
        contract_version=result.contract_version,
        warnings=result.warnings,
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
