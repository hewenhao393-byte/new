"""Display-only adapters for single-diagnosis service results."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

from pump_fault_app.presentation.status import (
    build_engineering_risk_level,
    build_runtime_processing_status,
)


_DISPLAY_LABELS = (
    ("正常", "正常"),
    ("转子不平衡", "转子不平衡"),
    ("联轴器不对中", "联轴器不对中"),
    ("松动", "机械松动"),
    ("轴承故障", "轴承故障"),
    ("汽蚀", "汽蚀"),
)


def build_probability_rows(top_probabilities: tuple[dict[str, float | str], ...]) -> list[dict[str, float | str]]:
    probability_map = {str(item["label"]): float(item["probability"]) for item in top_probabilities}
    return [
        {"label": display_label, "probability": round(probability_map.get(internal_label, 0.0), 6)}
        for internal_label, display_label in _DISPLAY_LABELS
    ]


def build_single_summary_items(
    *,
    quality_text: str,
    window_count: int | None,
    elapsed_seconds: float,
    message: str,
    warning_count: int,
) -> dict[str, str]:
    return {
        "信号质量": quality_text,
        "窗口数量": "-" if window_count is None else str(window_count),
        "推理耗时": f"{elapsed_seconds:.3f} s",
        "关键输出": message,
        "处理状态": build_runtime_processing_status(warning_count),
    }


def build_upload_signal_info(
    *,
    file_name: str,
    file_size_bytes: int | None,
    sampling_rate_hz: int,
    rpm: float,
    signal_column: str | None,
    time_column: str | None,
    measurement_position: str | None,
    device_id: str | None = None,
    vibration_direction: str | None = None,
) -> list[dict[str, str]]:
    size_text = "-" if file_size_bytes is None else _format_file_size(file_size_bytes)
    rows = [
        {"项目": "文件名称", "内容": file_name},
        {"项目": "文件大小", "内容": size_text},
        {"项目": "输入采样率", "内容": f"{sampling_rate_hz} Hz"},
        {"项目": "转速", "内容": f"{rpm:.1f} rpm"},
        {"项目": "信号列", "内容": signal_column or "自动识别"},
        {"项目": "时间列", "内容": time_column or "自动识别"},
        {"项目": "测点位置", "内容": measurement_position or "未填写"},
    ]
    if device_id is not None:
        rows.insert(2, {"项目": "设备编号", "内容": device_id or "未填写"})
    if vibration_direction is not None:
        rows.append({"项目": "振动方向", "内容": vibration_direction or "未填写"})
    return rows


def build_result_card_items(result: Any) -> dict[str, str]:
    """Format four engineering result indicators from the existing service output."""
    summary = result.summary
    label = getattr(summary, "diagnosis_label", None)
    confidence = getattr(summary, "confidence", None)
    predictions = getattr(result.inference_result, "window_predictions", None)
    consistency = None
    if predictions and label is not None:
        consistency = sum(1 for item in predictions if item.predicted_label == label) / len(predictions)
    return {
        "诊断结果": label or "-",
        "置信度": "-" if confidence is None else f"{float(confidence):.1%}",
        "窗口一致率": "-" if consistency is None else f"{consistency:.1%}",
        "风险等级": build_engineering_risk_level(
            success=bool(getattr(summary, "success", False)),
            label=label,
        ),
    }


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def build_single_visual_availability(result: Any) -> dict[str, bool]:
    inference_result = result.inference_result
    summary = result.summary
    visualization = result.visualization
    return {
        "probability_chart": bool(summary.top_probabilities),
        "window_distribution": bool(inference_result.window_predictions),
        "waveform": bool(visualization and visualization.time_domain is not None),
        "spectrum": bool(visualization and visualization.frequency_spectrum is not None),
        "envelope": bool(visualization and visualization.envelope_spectrum is not None),
        "wavelet": bool(visualization and visualization.wavelet_packet_energy is not None),
    }


def build_window_distribution_rows(window_predictions: tuple[Any, ...] | None) -> list[dict[str, str | int]]:
    if not window_predictions:
        return []
    counts = Counter(prediction.predicted_label for prediction in window_predictions)
    return [{"label": label, "count": count} for label, count in counts.items()]


def build_time_domain_rows(series: Any) -> pd.DataFrame:
    return pd.DataFrame({"time_seconds": list(series.time_s), "amplitude": list(series.amplitude)})


def build_spectrum_rows(series: Any) -> pd.DataFrame:
    return pd.DataFrame({"frequency_hz": list(series.frequency_hz), "amplitude": list(series.amplitude)})


def build_wavelet_packet_rows(series: Any) -> list[dict[str, float | str]]:
    return [{"band_label": label, "energy_ratio": ratio} for label, ratio in zip(series.band_labels, series.energy_ratio)]


def build_structured_summary_rows(summary: Any, inference_result: Any, diagnosis_time: str | None = None) -> list[dict[str, str]]:
    quality_text = "-"
    if inference_result.quality_report is not None:
        quality_text = inference_result.quality_report.quality_level
    timestamp = diagnosis_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        {"项目": "文件名称", "内容": str(summary.file_name or "-")},
        {"项目": "设备编号", "内容": str(summary.device_id or "-")},
        {"项目": "测点位置", "内容": str(summary.measurement_position or "-")},
        {"项目": "采样率", "内容": "-" if summary.sampling_rate_hz is None else f"{summary.sampling_rate_hz} Hz"},
        {"项目": "转速", "内容": "-" if summary.rpm is None else f"{summary.rpm:.1f} rpm"},
        {"项目": "预测类别", "内容": str(summary.diagnosis_label or "-")},
        {"项目": "综合置信度", "内容": "-" if summary.confidence is None else f"{summary.confidence:.3f}"},
        {"项目": "有效窗口数", "内容": "-" if summary.window_count is None else str(summary.window_count)},
        {"项目": "信号质量", "内容": quality_text},
        {"项目": "诊断时间", "内容": timestamp},
    ]
