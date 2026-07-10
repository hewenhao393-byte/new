from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

from pump_fault_app.domain.records import (
    DiagnosisVisualizationData,
    PreprocessedSignalRecord,
    RawSignalRecord,
    RecordPredictionResult,
    SignalQualityReport,
    WindowPredictionResult,
    WindowingResult,
)
from pump_fault_app.feature_extraction import extract_formal_features
from pump_fault_app.fusion import fuse_window_predictions
from pump_fault_app.inference.visualization import build_diagnosis_visualization
from pump_fault_app.io import RawSignalReadError, RawSignalReadRequest, read_vibration_signal
from pump_fault_app.prediction import load_formal_bp_bundle, predict_single_window
from pump_fault_app.preprocessing import preprocess_raw_signal
from pump_fault_app.quality import assess_signal_quality
from pump_fault_app.windowing import segment_preprocessed_signal


@dataclass(frozen=True)
class FormalInferenceRequest:
    file_path: Path
    sampling_rate_hz: int
    rpm: float
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None
    model_bundle_path: Path | None = None


@dataclass(frozen=True)
class FormalInferenceResult:
    success: bool
    failure_stage: str | None
    failure_message: str | None
    raw_signal: RawSignalRecord | None
    quality_report: SignalQualityReport | None
    preprocessed_signal: PreprocessedSignalRecord | None
    windowing_result: WindowingResult | None
    window_predictions: tuple[WindowPredictionResult, ...] | None
    record_prediction: RecordPredictionResult | None
    visualization: DiagnosisVisualizationData | None
    runtime_warnings: tuple[str, ...] = ()


def run_formal_inference(request: FormalInferenceRequest) -> FormalInferenceResult:
    raw_signal: RawSignalRecord | None = None
    quality_report: SignalQualityReport | None = None
    preprocessed_signal: PreprocessedSignalRecord | None = None
    windowing_result: WindowingResult | None = None
    window_predictions: tuple[WindowPredictionResult, ...] | None = None

    try:
        raw_signal = read_vibration_signal(
            RawSignalReadRequest(
                file_path=request.file_path,
                sampling_rate_hz=request.sampling_rate_hz,
                rpm=request.rpm,
                signal_column=request.signal_column,
                time_column=request.time_column,
                device_id=request.device_id,
                measurement_position=request.measurement_position,
            )
        )
    except RawSignalReadError as exc:
        return FormalInferenceResult(
            success=False,
            failure_stage="input",
            failure_message=str(exc),
            raw_signal=None,
            quality_report=None,
            preprocessed_signal=None,
            windowing_result=None,
            window_predictions=None,
            record_prediction=None,
            visualization=None,
            runtime_warnings=(),
        )

    quality_report = assess_signal_quality(raw_signal)
    if not quality_report.allow_diagnosis:
        return FormalInferenceResult(
            success=False,
            failure_stage="quality",
            failure_message="; ".join(quality_report.rejection_reasons),
            raw_signal=raw_signal,
            quality_report=quality_report,
            preprocessed_signal=None,
            windowing_result=None,
            window_predictions=None,
            record_prediction=None,
            visualization=None,
            runtime_warnings=(),
        )

    preprocessed_signal = preprocess_raw_signal(raw_signal)
    windowing_result = segment_preprocessed_signal(preprocessed_signal)
    bundle = load_formal_bp_bundle(request.model_bundle_path)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        window_predictions = tuple(
            predict_single_window(extract_formal_features(window), bundle)
            for window in windowing_result.windows
        )
    record_prediction = fuse_window_predictions(
        window_predictions,
        source_file=raw_signal.source_file,
    )
    runtime_warnings = tuple(str(item.message) for item in caught_warnings)
    try:
        visualization = build_diagnosis_visualization(
            raw_signal=raw_signal,
            preprocessed_signal=preprocessed_signal,
            window_predictions=window_predictions,
        )
    except Exception as exc:
        visualization = DiagnosisVisualizationData(
            time_domain=None,
            frequency_spectrum=None,
            envelope_spectrum=None,
            wavelet_packet_energy=None,
            warnings=(f"visualization generation failed: {exc}",),
        )
    return FormalInferenceResult(
        success=True,
        failure_stage=None,
        failure_message=None,
        raw_signal=raw_signal,
        quality_report=quality_report,
        preprocessed_signal=preprocessed_signal,
        windowing_result=windowing_result,
        window_predictions=window_predictions,
        record_prediction=record_prediction,
        visualization=visualization,
        runtime_warnings=runtime_warnings,
    )
