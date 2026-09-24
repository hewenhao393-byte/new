from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from pump_fault_app.domain.formal_contract import (
    FORMAL_LABEL_ORDER,
    FORMAL_V3_CONTRACT,
)
from pump_fault_app.presentation.status import (
    build_engineering_risk_level,
    build_runtime_processing_status,
    build_user_runtime_notice,
)
from pump_fault_app.version import APP_VERSION, MODEL_VERSION


@dataclass(frozen=True)
class ReportKeyValueItem:
    label: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True)
class ReportProbabilityItem:
    label: str
    probability: float
    percentage_text: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "label": self.label,
            "probability": float(self.probability),
            "probability_text": self.percentage_text,
        }


@dataclass(frozen=True)
class ReportWindowDistributionItem:
    label: str
    window_count: int
    ratio: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "label": self.label,
            "count": int(self.window_count),
            "ratio": float(self.ratio),
            "ratio_text": f"{float(self.ratio):.1%}",
        }


@dataclass(frozen=True)
class ReportVisualizationAvailability:
    time_domain: bool
    frequency_spectrum: bool
    envelope_spectrum: bool
    wavelet_packet_energy: bool
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, dict[str, bool | str]]:
        message_map = {
            "time_domain": "时域波形数据不可用",
            "frequency_spectrum": "频谱图数据不可用",
            "envelope_spectrum": "包络谱数据不可用",
            "wavelet_packet_energy": "小波包能量图数据不可用",
        }
        values = {
            "time_domain": self.time_domain,
            "frequency_spectrum": self.frequency_spectrum,
            "envelope_spectrum": self.envelope_spectrum,
            "wavelet_packet_energy": self.wavelet_packet_energy,
        }
        return {
            key: {
                "available": bool(available),
                "message": "" if available else message_map[key],
            }
            for key, available in values.items()
        }


@dataclass(frozen=True)
class SingleReportConclusion:
    diagnosis_label: str
    top_probability: str
    second_label: str
    probability_gap: str
    window_consistency: str
    diagnosis_grade: str
    risk_level: str
    diagnosis_advice: str
    warning_messages: tuple[str, ...]

    def to_dict(
        self,
        *,
        second_probability: float | None = None,
        runtime_alert_count: int = 0,
    ) -> dict[str, str | float | int | None | list[str]]:
        confidence_value = _parse_decimal_text(self.top_probability)
        margin_value = _parse_decimal_text(self.probability_gap)
        consistency_value = _parse_percent_text(self.window_consistency)
        return {
            "final_label": self.diagnosis_label,
            "confidence": confidence_value,
            "confidence_text": _format_percent(confidence_value),
            "second_label": self.second_label,
            "second_probability": second_probability,
            "second_probability_text": _format_percent(second_probability),
            "probability_margin": margin_value,
            "probability_margin_text": _format_percent(margin_value),
            "window_consistency": consistency_value,
            "window_consistency_text": self.window_consistency if consistency_value is not None else "-",
            "diagnosis_level": self.diagnosis_grade,
            "risk_level": self.risk_level,
            "suggestion": self.diagnosis_advice,
            "warnings": list(self.warning_messages),
            "runtime_alert_count": int(runtime_alert_count),
        }


@dataclass(frozen=True)
class SingleReportViewData:
    basic_info: tuple[ReportKeyValueItem, ...]
    conclusion: SingleReportConclusion
    probabilities: tuple[ReportProbabilityItem, ...]
    window_distribution: tuple[ReportWindowDistributionItem, ...]
    processing_parameters: tuple[ReportKeyValueItem, ...]
    method_steps: tuple[str, ...]
    visualization_availability: ReportVisualizationAvailability
    time_domain: Any | None
    frequency_spectrum: Any | None
    envelope_spectrum: Any | None
    wavelet_packet_energy: Any | None
    quality_text: str
    summary_items: tuple[ReportKeyValueItem, ...]
    runtime_alert_count: int

    def to_dict(self, *, include_visualization_data: bool = False) -> dict[str, Any]:
        second_probability = next(
            (float(item.probability) for item in self.probabilities if item.label == self.conclusion.second_label),
            None,
        )
        payload: dict[str, Any] = {
            "basic_info": [item.to_dict() for item in self.basic_info],
            "conclusion": self.conclusion.to_dict(
                second_probability=second_probability,
                runtime_alert_count=self.runtime_alert_count,
            ),
            "class_probabilities": [item.to_dict() for item in self.probabilities],
            "window_distribution": [item.to_dict() for item in self.window_distribution],
            "processing_parameters": [item.to_dict() for item in self.processing_parameters],
            "method_steps": list(self.method_steps),
            "visualization_availability": self.visualization_availability.to_dict(),
            "visualization_summary": {
                "time_domain_points": _series_point_count(self.time_domain, "time_s"),
                "frequency_spectrum_points": _series_point_count(self.frequency_spectrum, "frequency_hz"),
                "envelope_spectrum_points": _series_point_count(self.envelope_spectrum, "frequency_hz"),
                "wavelet_packet_energy_bands": _series_point_count(self.wavelet_packet_energy, "band_labels"),
            },
            "quality_text": self.quality_text,
            "summary_items": [item.to_dict() for item in self.summary_items],
        }
        if include_visualization_data:
            payload["visualization_data"] = {
                "time_domain": _serialize_time_domain(self.time_domain),
                "frequency_spectrum": _serialize_spectrum(self.frequency_spectrum),
                "envelope_spectrum": _serialize_spectrum(self.envelope_spectrum),
                "wavelet_packet_energy": _serialize_wavelet_packet_energy(self.wavelet_packet_energy),
            }
        return payload


def build_single_report_view_data(result: Any) -> SingleReportViewData:
    summary = result.summary
    inference_result = result.inference_result
    raw_signal = getattr(inference_result, "raw_signal", None)
    preprocessed_signal = getattr(inference_result, "preprocessed_signal", None)
    quality_report = getattr(inference_result, "quality_report", None)
    quality_text = quality_report.quality_level if quality_report is not None else "-"

    target_sampling_rate = "-"
    if preprocessed_signal is not None:
        target_sampling_rate = f"{preprocessed_signal.target_sampling_rate_hz} Hz"
    elif getattr(summary, "sampling_rate_hz", None) is not None:
        target_sampling_rate = f"{FORMAL_V3_CONTRACT.target_sampling_rate} Hz"

    signal_duration = "-"
    if raw_signal is not None and getattr(raw_signal, "duration_seconds", None) is not None:
        signal_duration = f"{raw_signal.duration_seconds:.3f} s"

    visualization_availability = _build_visualization_availability(result)
    summary_top_probabilities = getattr(summary, "top_probabilities", ())
    summary_confidence = getattr(summary, "confidence", None)
    top_probability, second_label, second_probability = _top_two_probabilities(summary_top_probabilities, summary_confidence)
    probability_gap = "-" if summary_confidence is None else f"{(top_probability - second_probability):.3f}"
    window_consistency = _compute_window_consistency(result)

    runtime_alert_count = len(getattr(summary, "runtime_alerts", ()))
    runtime_notice = build_user_runtime_notice(runtime_alert_count)
    warning_messages = tuple(
        message
        for message in (runtime_notice, *visualization_availability.messages)
        if message
    )
    diagnosis_label = getattr(summary, "diagnosis_label", None)
    success = bool(getattr(summary, "success", diagnosis_label is not None))
    return SingleReportViewData(
        basic_info=(
            ReportKeyValueItem("文件名", str(getattr(summary, "file_name", None) or "-")),
            ReportKeyValueItem("设备编号", str(getattr(summary, "device_id", None) or "-")),
            ReportKeyValueItem("测点位置", str(getattr(summary, "measurement_position", None) or "-")),
            ReportKeyValueItem("振动方向", str(getattr(result, "vibration_direction", None) or "-")),
            ReportKeyValueItem("原始采样率", "-" if getattr(summary, "sampling_rate_hz", None) is None else f"{summary.sampling_rate_hz} Hz"),
            ReportKeyValueItem("目标采样率", target_sampling_rate),
            ReportKeyValueItem("转速", "-" if getattr(summary, "rpm", None) is None else f"{summary.rpm:.1f} rpm"),
            ReportKeyValueItem("信号时长", signal_duration),
            ReportKeyValueItem("窗口数量", "-" if getattr(summary, "window_count", None) is None else str(summary.window_count)),
            ReportKeyValueItem("模型版本", MODEL_VERSION),
            ReportKeyValueItem("软件版本", APP_VERSION),
        ),
        conclusion=SingleReportConclusion(
            diagnosis_label=getattr(summary, "diagnosis_label", None) or "-",
            top_probability="-" if getattr(summary, "confidence", None) is None else f"{top_probability:.3f}",
            second_label=second_label,
            probability_gap=probability_gap,
            window_consistency="-" if window_consistency is None else f"{window_consistency:.1%}",
            diagnosis_grade=_build_diagnosis_grade(getattr(summary, "confidence", None)),
            risk_level=build_engineering_risk_level(success=success, label=diagnosis_label),
            diagnosis_advice=_build_diagnosis_advice(diagnosis_label),
            warning_messages=warning_messages,
        ),
        probabilities=_build_probability_items(summary_top_probabilities),
        window_distribution=_build_window_distribution(result),
        processing_parameters=_build_processing_parameters(),
        method_steps=(
            "读取振动信号及设备工况信息",
            "检查信号长度、有限值与基本质量",
            "去除直流分量并统一重采样至12000 Hz",
            "采用5～5000 Hz零相位带通滤波",
            "按4800点窗口、2400点步长进行滑动分段",
            "按冻结顺序提取43维振动特征",
            "调用通道独立 CatBoost 完成窗口级六分类并融合概率",
            "自动生成诊断结果与报告",
        ),
        visualization_availability=visualization_availability,
        time_domain=None if result.visualization is None else result.visualization.time_domain,
        frequency_spectrum=None if result.visualization is None else result.visualization.frequency_spectrum,
        envelope_spectrum=None if result.visualization is None else result.visualization.envelope_spectrum,
        wavelet_packet_energy=None if result.visualization is None else result.visualization.wavelet_packet_energy,
        quality_text=quality_text,
        summary_items=(
            ReportKeyValueItem("信号质量", quality_text),
            ReportKeyValueItem("窗口数量", "-" if getattr(summary, "window_count", None) is None else str(summary.window_count)),
            ReportKeyValueItem("关键输出", str(getattr(summary, "message", "-"))),
            ReportKeyValueItem("处理状态", build_runtime_processing_status(runtime_alert_count)),
        ),
        runtime_alert_count=runtime_alert_count,
    )


def _build_probability_items(top_probabilities: tuple[dict[str, float | str], ...]) -> tuple[ReportProbabilityItem, ...]:
    probability_map = {str(item["label"]): float(item["probability"]) for item in top_probabilities}
    return tuple(
        ReportProbabilityItem(
            label=label,
            probability=probability_map.get(label, 0.0),
            percentage_text=f"{probability_map.get(label, 0.0):.1%}",
        )
        for label in FORMAL_LABEL_ORDER
    )


def _build_window_distribution(result: Any) -> tuple[ReportWindowDistributionItem, ...]:
    predictions = getattr(result.inference_result, "window_predictions", None)
    if not predictions:
        return ()
    counts = Counter(item.predicted_label for item in predictions)
    total = len(predictions)
    return tuple(
        ReportWindowDistributionItem(
            label=label,
            window_count=int(counts[label]),
            ratio=float(counts[label] / total),
        )
        for label in FORMAL_LABEL_ORDER
        if counts.get(label, 0) > 0
    )


def _build_visualization_availability(result: Any) -> ReportVisualizationAvailability:
    visualization = result.visualization
    flags = {
        "time_domain": bool(visualization and visualization.time_domain is not None),
        "frequency_spectrum": bool(visualization and visualization.frequency_spectrum is not None),
        "envelope_spectrum": bool(visualization and visualization.envelope_spectrum is not None),
        "wavelet_packet_energy": bool(visualization and visualization.wavelet_packet_energy is not None),
    }
    messages: list[str] = []
    if not any(flags.values()):
        messages.append("暂无可视化数据，报告页仅展示正式诊断结论。")
    else:
        if not flags["time_domain"]:
            messages.append("时域波形数据不可用")
        if not flags["frequency_spectrum"]:
            messages.append("频谱图数据不可用")
        if not flags["envelope_spectrum"]:
            messages.append("包络谱数据不可用")
        if not flags["wavelet_packet_energy"]:
            messages.append("小波包能量图数据不可用")
    if visualization is not None and visualization.warnings:
        messages.append("部分振动特征图未生成，正式诊断结果不受影响。")
    return ReportVisualizationAvailability(
        time_domain=flags["time_domain"],
        frequency_spectrum=flags["frequency_spectrum"],
        envelope_spectrum=flags["envelope_spectrum"],
        wavelet_packet_energy=flags["wavelet_packet_energy"],
        messages=tuple(messages),
    )


def _build_processing_parameters() -> tuple[ReportKeyValueItem, ...]:
    contract = FORMAL_V3_CONTRACT
    return (
        ReportKeyValueItem("统一采样率", f"{contract.target_sampling_rate} Hz"),
        ReportKeyValueItem("分析频带", f"{contract.filter_low_hz:g}～{contract.filter_high_hz:g} Hz"),
        ReportKeyValueItem("窗口长度", f"{contract.window_size}点（0.2 s）"),
        ReportKeyValueItem("窗口步长", f"{contract.step_size}点（50%重叠）"),
        ReportKeyValueItem("窗口与步长", f"{contract.window_size}点 / {contract.step_size}点"),
        ReportKeyValueItem("小波包参数", f"{contract.wavelet}，{contract.wavelet_level}层分解"),
        ReportKeyValueItem(
            "包络分析频带",
            f"{contract.envelope_low_hz:g}～{contract.envelope_high_hz:g} Hz",
        ),
        ReportKeyValueItem("模型输入", f"{len(contract.feature_names)}维振动特征"),
    )


def _top_two_probabilities(
    top_probabilities: tuple[dict[str, float | str], ...],
    confidence: float | None,
) -> tuple[float, str, float]:
    top_probability = 0.0 if confidence is None else float(confidence)
    second_label = "-"
    second_probability = 0.0
    if len(top_probabilities) >= 2:
        second_label = str(top_probabilities[1]["label"])
        second_probability = float(top_probabilities[1]["probability"])
    return top_probability, second_label, second_probability


def _compute_window_consistency(result: Any) -> float | None:
    predictions = getattr(result.inference_result, "window_predictions", None)
    diagnosis_label = getattr(result.summary, "diagnosis_label", None)
    if not predictions or diagnosis_label is None:
        return None
    match_count = sum(1 for item in predictions if item.predicted_label == diagnosis_label)
    return match_count / len(predictions)


def _build_diagnosis_grade(confidence: float | None) -> str:
    if confidence is None:
        return "-"
    if confidence >= 0.85:
        return "高"
    if confidence >= 0.60:
        return "中"
    return "低"


def _build_diagnosis_advice(label: str | None) -> str:
    if label == "正常":
        return "建议继续按计划监测。"
    if label is None:
        return "建议先核查输入记录与信号质量。"
    return "建议结合设备工况与现场检修记录进一步复核。"


def _parse_decimal_text(value: str) -> float | None:
    if value == "-":
        return None
    return float(value)


def _parse_percent_text(value: str) -> float | None:
    if value == "-":
        return None
    return float(value.rstrip("%")) / 100.0


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1%}"


def _series_point_count(series: Any | None, field_name: str) -> int | None:
    if series is None:
        return None
    values = getattr(series, field_name, None)
    if values is None:
        return None
    return int(len(values))


def _serialize_time_domain(series: Any | None) -> dict[str, Any] | None:
    if series is None:
        return None
    return {
        "time_s": [float(item) for item in series.time_s],
        "amplitude": [float(item) for item in series.amplitude],
        "sampling_rate_hz": int(series.sampling_rate_hz),
        "point_count": int(series.point_count),
        "downsampled_for_display": bool(series.downsampled_for_display),
    }


def _serialize_spectrum(series: Any | None) -> dict[str, Any] | None:
    if series is None:
        return None
    return {
        "frequency_hz": [float(item) for item in series.frequency_hz],
        "amplitude": [float(item) for item in series.amplitude],
        "frequency_min_hz": float(series.frequency_min_hz),
        "frequency_max_hz": float(series.frequency_max_hz),
        "resolution_hz": float(series.resolution_hz),
    }


def _serialize_wavelet_packet_energy(series: Any | None) -> dict[str, Any] | None:
    if series is None:
        return None
    return {
        "band_labels": [str(item) for item in series.band_labels],
        "band_start_hz": [float(item) for item in series.band_start_hz],
        "band_end_hz": [float(item) for item in series.band_end_hz],
        "energy_ratio": [float(item) for item in series.energy_ratio],
        "wavelet": str(series.wavelet),
        "decomposition_level": int(series.decomposition_level),
    }
