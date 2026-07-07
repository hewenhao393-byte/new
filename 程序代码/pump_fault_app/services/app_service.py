from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pump_fault_app.batch import (
    BatchInferenceRequest,
    BatchInferenceResult,
    load_batch_manifest,
    run_batch_inference,
    run_batch_inference_from_manifest,
)
from pump_fault_app.export import (
    BatchExportResult,
    SummaryExportResult,
    create_export_output_dir,
    export_batch_result,
    export_diagnosis_summary,
)
from pump_fault_app.inference import FormalInferenceRequest, FormalInferenceResult, run_formal_inference
from pump_fault_app.reporting import DiagnosisSummary, build_diagnosis_summary


@dataclass(frozen=True)
class AppSingleRunRequest:
    file_path: Path
    sampling_rate_hz: int
    rpm: float
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None
    model_bundle_path: Path | None = None
    export_root: Path | None = None


@dataclass(frozen=True)
class AppSingleRunResult:
    inference_result: FormalInferenceResult
    summary: DiagnosisSummary
    export_result: SummaryExportResult | None


@dataclass(frozen=True)
class AppBatchRunRequest:
    file_paths: tuple[Path, ...] | None = None
    manifest_path: Path | None = None
    sampling_rate_hz: int | None = None
    rpm: float | None = None
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None
    model_bundle_path: Path | None = None
    export_root: Path | None = None


@dataclass(frozen=True)
class AppBatchRunResult:
    batch_result: BatchInferenceResult
    export_result: BatchExportResult | None


def run_single_diagnosis(request: AppSingleRunRequest) -> AppSingleRunResult:
    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=request.file_path,
            sampling_rate_hz=request.sampling_rate_hz,
            rpm=request.rpm,
            signal_column=request.signal_column,
            time_column=request.time_column,
            device_id=request.device_id,
            measurement_position=request.measurement_position,
            model_bundle_path=request.model_bundle_path,
        )
    )
    summary = build_diagnosis_summary(inference_result)
    export_result = None
    if request.export_root is not None:
        export_result = export_diagnosis_summary(
            summary,
            output_dir=create_export_output_dir(request.export_root, prefix="single_run"),
        )
    return AppSingleRunResult(
        inference_result=inference_result,
        summary=summary,
        export_result=export_result,
    )


def run_batch_diagnosis(request: AppBatchRunRequest) -> AppBatchRunResult:
    if bool(request.file_paths) == bool(request.manifest_path):
        raise ValueError("exactly one of file_paths or manifest_path must be provided")

    if request.manifest_path is not None:
        batch_result = run_batch_inference_from_manifest(
            load_batch_manifest(request.manifest_path),
            model_bundle_path=request.model_bundle_path,
        )
    else:
        if request.sampling_rate_hz is None or request.rpm is None:
            raise ValueError("sampling_rate_hz and rpm are required when using file_paths")
        batch_result = run_batch_inference(
            BatchInferenceRequest(
                file_paths=tuple(request.file_paths or ()),
                sampling_rate_hz=request.sampling_rate_hz,
                rpm=request.rpm,
                signal_column=request.signal_column,
                time_column=request.time_column,
                device_id=request.device_id,
                measurement_position=request.measurement_position,
                model_bundle_path=request.model_bundle_path,
            )
        )

    export_result = None
    if request.export_root is not None:
        export_result = export_batch_result(
            batch_result,
            output_dir=create_export_output_dir(request.export_root, prefix="batch_run"),
        )
    return AppBatchRunResult(batch_result=batch_result, export_result=export_result)
