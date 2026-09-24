from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult, MultiChannelInferenceRequest
from pump_fault_app.reporting.summary import DiagnosisSummary, build_diagnosis_summary


@dataclass(frozen=True)
class SingleReportViewData:
    result: MultiChannelDiagnosisResult
    summary: DiagnosisSummary
    source_files: dict[str, str]
    sampling_rate_hz: int | None
    rpm: float | None
    vibration_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.as_dict(),
            "source_files": dict(self.source_files),
            "sampling_rate_hz": self.sampling_rate_hz,
            "rpm": self.rpm,
            "vibration_direction": self.vibration_direction,
            "channel_results": [
                {
                    "channel": item.channel,
                    "status": item.status,
                    "failure_stage": item.failure_stage,
                    "failure_message": item.failure_message,
                    "predicted_label": item.predicted_label,
                    "class_probabilities": item.class_probabilities,
                    "window_count": item.window_count,
                    "window_consistency": item.window_consistency,
                    "warnings": item.warnings,
                }
                for item in self.result.channel_results
            ],
        }


def build_single_report_view_data(value: Any) -> SingleReportViewData:
    if isinstance(value, MultiChannelDiagnosisResult):
        result = value
        request = None
        vibration_direction = None
    else:
        result = value.inference_result
        request = getattr(value, "inference_request", None)
        vibration_direction = getattr(value, "vibration_direction", None)
    if not isinstance(result, MultiChannelDiagnosisResult):
        raise TypeError("report input must contain a MultiChannelDiagnosisResult")
    if request is not None and not isinstance(request, MultiChannelInferenceRequest):
        raise TypeError("report inference_request must be MultiChannelInferenceRequest")
    source_files = (
        {item.channel: str(Path(item.file_path)) for item in request.channels}
        if request is not None
        else {channel: "-" for channel in result.input_channels}
    )
    return SingleReportViewData(
        result=result,
        summary=build_diagnosis_summary(result),
        source_files=source_files,
        sampling_rate_hz=request.sampling_rate_hz if request is not None else None,
        rpm=request.rpm if request is not None else None,
        vibration_direction=vibration_direction,
    )
