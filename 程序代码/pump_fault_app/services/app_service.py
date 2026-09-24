from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pump_fault_app.batch import (
    BatchInferenceRequest,
    BatchInferenceResult,
    load_batch_manifest,
    run_batch_inference,
    run_batch_inference_from_manifest,
)
from pump_fault_app.domain.records import DiagnosisVisualizationData
from pump_fault_app.history.models import DiagnosisHistoryRecord
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
    vibration_direction: str | None = None
    model_bundle_path: Path | None = None
    export_root: Path | None = None
    history_database_path: Path | None = None
    history_report_dir: Path | None = None
    sample_database_path: Path | None = None


@dataclass(frozen=True)
class AppSingleRunResult:
    inference_result: FormalInferenceResult
    summary: DiagnosisSummary
    export_result: SummaryExportResult | None
    visualization: DiagnosisVisualizationData | None
    vibration_direction: str | None = None
    history_record: DiagnosisHistoryRecord | None = None
    history_warning: str | None = None


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
    app_result = AppSingleRunResult(
        inference_result=inference_result,
        summary=summary,
        export_result=export_result,
        visualization=inference_result.visualization,
        vibration_direction=request.vibration_direction,
    )
    if not summary.success:
        return app_result

    try:
        from pump_fault_app.history.service import (
            DEFAULT_HISTORY_DATABASE_PATH,
            DEFAULT_HISTORY_REPORT_DIR,
            record_single_diagnosis_history,
        )
        from pump_fault_app.sample_repository.service import DEFAULT_SAMPLE_DATABASE_PATH

        history_record = record_single_diagnosis_history(
            app_result,
            database_path=request.history_database_path or DEFAULT_HISTORY_DATABASE_PATH,
            report_dir=request.history_report_dir or DEFAULT_HISTORY_REPORT_DIR,
            sample_database_path=(
                request.sample_database_path
                if request.sample_database_path is not None
                else DEFAULT_SAMPLE_DATABASE_PATH
            ),
        )
    except Exception:
        return replace(
            app_result,
            history_warning="诊断已完成，但历史记录或Word报告未能保存。",
        )
    return replace(app_result, history_record=history_record)


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
