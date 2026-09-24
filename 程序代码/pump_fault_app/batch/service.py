from __future__ import annotations

from dataclasses import dataclass
from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult, MultiChannelInferenceRequest
from pump_fault_app.inference import run_multichannel_inference


@dataclass(frozen=True)
class BatchInferenceRequest:
    items: tuple[MultiChannelInferenceRequest, ...]


@dataclass(frozen=True)
class BatchInferenceResult:
    total_count: int
    success_count: int
    failure_count: int
    diagnosed_count: int
    results: tuple[MultiChannelDiagnosisResult, ...]


def run_batch_inference(request: BatchInferenceRequest) -> BatchInferenceResult:
    if not request.items:
        raise ValueError("at least one multichannel request is required")
    results = tuple(run_multichannel_inference(item) for item in request.items)
    diagnosed_count = sum(item.status == "diagnosed" for item in results)
    success_count = diagnosed_count
    total_count = len(results)
    return BatchInferenceResult(
        total_count=total_count,
        success_count=success_count,
        failure_count=total_count - success_count,
        diagnosed_count=diagnosed_count,
        results=results,
    )
