from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference
from pump_fault_app.reporting import DiagnosisSummary, build_diagnosis_summary


@dataclass(frozen=True)
class BatchInferenceRequest:
    file_paths: tuple[Path, ...]
    sampling_rate_hz: int
    rpm: float
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None
    model_bundle_path: Path | None = None


@dataclass(frozen=True)
class BatchInferenceResult:
    total_count: int
    success_count: int
    failure_count: int
    diagnosed_count: int
    rejected_count: int
    input_error_count: int
    summaries: tuple[DiagnosisSummary, ...]


def run_batch_inference(request: BatchInferenceRequest) -> BatchInferenceResult:
    if not request.file_paths:
        raise ValueError("at least one file path is required")

    summaries = tuple(
        build_diagnosis_summary(
            run_formal_inference(
                FormalInferenceRequest(
                    file_path=path,
                    sampling_rate_hz=request.sampling_rate_hz,
                    rpm=request.rpm,
                    signal_column=request.signal_column,
                    time_column=request.time_column,
                    device_id=request.device_id,
                    measurement_position=request.measurement_position,
                    model_bundle_path=request.model_bundle_path,
                )
            )
        )
        for path in request.file_paths
    )
    diagnosed_count = sum(summary.status == "diagnosed" for summary in summaries)
    rejected_count = sum(summary.status == "rejected" for summary in summaries)
    input_error_count = sum(summary.status == "input_error" for summary in summaries)
    success_count = sum(summary.success for summary in summaries)
    total_count = len(summaries)
    return BatchInferenceResult(
        total_count=total_count,
        success_count=success_count,
        failure_count=total_count - success_count,
        diagnosed_count=diagnosed_count,
        rejected_count=rejected_count,
        input_error_count=input_error_count,
        summaries=summaries,
    )
