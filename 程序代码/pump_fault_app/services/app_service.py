from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pump_fault_app.batch import (
    BatchInferenceResult,
    load_batch_manifest,
    run_batch_inference_from_manifest,
)
from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult, MultiChannelInferenceRequest
from pump_fault_app.inference import run_multichannel_inference


@dataclass(frozen=True)
class AppSingleRunRequest:
    inference_request: MultiChannelInferenceRequest
    vibration_direction: str | None = None
    export_root: Path | None = None
    history_database_path: Path | None = None
    history_report_dir: Path | None = None


@dataclass(frozen=True)
class AppSingleRunResult:
    inference_result: MultiChannelDiagnosisResult
    vibration_direction: str | None = None
    history_record: object | None = None
    history_warning: str | None = None


@dataclass(frozen=True)
class AppBatchRunRequest:
    items: tuple[MultiChannelInferenceRequest, ...] | None = None
    manifest_path: Path | None = None
    model_directory: Path | None = None


@dataclass(frozen=True)
class AppBatchRunResult:
    batch_result: BatchInferenceResult


def run_single_diagnosis(request: AppSingleRunRequest) -> AppSingleRunResult:
    app_result = AppSingleRunResult(
        inference_result=run_multichannel_inference(request.inference_request),
        vibration_direction=request.vibration_direction,
    )
    if app_result.inference_result.status != "diagnosed":
        return app_result
    try:
        from pump_fault_app.history.service import (
            DEFAULT_HISTORY_DATABASE_PATH,
            record_single_diagnosis_history,
        )

        history_record = record_single_diagnosis_history(
            app_result.inference_result,
            source_files={item.channel: str(item.file_path) for item in request.inference_request.channels},
            sampling_rate_hz=request.inference_request.sampling_rate_hz,
            rpm=request.inference_request.rpm,
            database_path=request.history_database_path or DEFAULT_HISTORY_DATABASE_PATH,
        )
        return replace(app_result, history_record=history_record)
    except Exception:
        return replace(app_result, history_warning="诊断已完成，但V3历史记录未能保存。")


def run_batch_diagnosis(request: AppBatchRunRequest) -> AppBatchRunResult:
    if bool(request.items) == bool(request.manifest_path):
        raise ValueError("exactly one of items or manifest_path must be provided")

    if request.manifest_path is not None:
        batch_result = run_batch_inference_from_manifest(
            load_batch_manifest(request.manifest_path),
            model_directory=request.model_directory,
        )
    else:
        results = tuple(run_multichannel_inference(item) for item in request.items or ())
        diagnosed_count = sum(item.status == "diagnosed" for item in results)
        batch_result = BatchInferenceResult(
            total_count=len(results),
            success_count=diagnosed_count,
            failure_count=len(results) - diagnosed_count,
            diagnosed_count=diagnosed_count,
            results=results,
        )
    return AppBatchRunResult(batch_result=batch_result)
