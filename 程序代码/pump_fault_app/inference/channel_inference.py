from __future__ import annotations

from typing import Any

from pump_fault_app.domain.diagnosis_models import (
    ChannelDiagnosisResult,
    ChannelInput,
    MultiChannelInferenceRequest,
)
from pump_fault_app.domain.records import RawSignalRecord
from pump_fault_app.feature_extraction import extract_catboost43_features
from pump_fault_app.fusion import fuse_window_probabilities
from pump_fault_app.prediction import predict_channel_window
from pump_fault_app.preprocessing import preprocess_raw_signal
from pump_fault_app.quality import assess_signal_quality
from pump_fault_app.windowing import segment_preprocessed_signal
from pump_fault_app.inference.visualization import build_diagnosis_visualization


def _invalid(
    item: ChannelInput,
    stage: str,
    exc: Exception | str,
    *,
    quality_report: Any | None = None,
) -> ChannelDiagnosisResult:
    message = str(exc)
    return ChannelDiagnosisResult(
        channel=item.channel,
        status="invalid",
        failure_stage=stage,
        failure_message=message,
        quality_report=quality_report,
        warnings=(message,),
    )


def run_channel_inference(
    item: ChannelInput,
    raw_signal: RawSignalRecord,
    request: MultiChannelInferenceRequest,
    model: Any,
) -> ChannelDiagnosisResult:
    try:
        quality_report = assess_signal_quality(raw_signal)
    except Exception as exc:
        return _invalid(item, "quality", exc)
    if not quality_report.allow_diagnosis:
        return _invalid(
            item,
            "quality",
            "; ".join(quality_report.rejection_reasons),
            quality_report=quality_report,
        )

    try:
        preprocessed = preprocess_raw_signal(raw_signal)
    except Exception as exc:
        return _invalid(item, "preprocessing", exc, quality_report=quality_report)
    try:
        windowing = segment_preprocessed_signal(preprocessed)
        if not windowing.windows:
            raise ValueError("no complete 4800-sample window is available")
    except Exception as exc:
        return _invalid(item, "windowing", exc, quality_report=quality_report)

    predictions = []
    for window in windowing.windows:
        try:
            feature_vector = extract_catboost43_features(window.samples, request.rpm)
        except Exception as exc:
            return _invalid(item, "feature_extraction", exc, quality_report=quality_report)
        try:
            predictions.append(
                predict_channel_window(
                    feature_vector,
                    model,
                    item.channel,
                    window_index=window.window_index,
                    start_index=window.start_index,
                    end_index=window.end_index,
                )
            )
        except Exception as exc:
            return _invalid(item, "prediction", exc, quality_report=quality_report)

    visualization = build_diagnosis_visualization(
        raw_signal=raw_signal,
        preprocessed_signal=preprocessed,
        window_predictions=None,
    )
    try:
        return fuse_window_probabilities(
            item.channel,
            predictions,
            quality_report=quality_report,
            visualization=visualization,
            warnings=tuple(quality_report.warnings) + tuple(visualization.warnings),
        )
    except Exception as exc:
        return _invalid(item, "window_fusion", exc, quality_report=quality_report)
