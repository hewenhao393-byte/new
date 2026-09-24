from __future__ import annotations

from dataclasses import dataclass
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
    return AppSingleRunResult(
        inference_result=run_multichannel_inference(request.inference_request),
        vibration_direction=request.vibration_direction,
    )


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
