"""Display-only adapters for batch diagnosis results."""

from __future__ import annotations

from typing import Any, Iterable


_STATUS_LABELS = {
    "diagnosed": "已完成",
    "rejected": "质量拒绝",
    "input_error": "输入错误",
}


def build_batch_filter_options(summaries: Iterable[dict[str, Any]]) -> list[str]:
    labels = sorted({str(summary["diagnosis_label"]) for summary in summaries if summary.get("diagnosis_label")})
    return ["全部", *labels]


def build_batch_table_rows(summaries: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        confidence = summary.get("confidence")
        status = str(summary.get("status") or "")
        rows.append(
            {
                "文件名": str(summary.get("file_name") or "-"),
                "设备编号": str(summary.get("device_id") or "-"),
                "预测类别": str(summary.get("diagnosis_label") or "-"),
                "置信度": "-" if confidence is None else f"{float(confidence):.1%}",
                "信号质量": str(summary.get("signal_quality") or _infer_signal_quality(status)),
                "状态": _STATUS_LABELS.get(status, "-"),
            }
        )
    return rows


def build_batch_task_statistics(summaries: Iterable[dict[str, Any]]) -> dict[str, str]:
    items = list(summaries)
    diagnosed = [item for item in items if item.get("status") == "diagnosed"]
    abnormal_count = sum(1 for item in diagnosed if item.get("diagnosis_label") not in {None, "正常"})
    confidences = [float(item["confidence"]) for item in diagnosed if item.get("confidence") is not None]
    return {
        "总文件数": str(len(items)),
        "完成数量": str(len(diagnosed)),
        "异常数量": str(abnormal_count),
        "平均置信度": "-" if not confidences else f"{sum(confidences) / len(confidences):.1%}",
    }


def _infer_signal_quality(status: str) -> str:
    if status == "diagnosed":
        return "pass"
    if status == "rejected":
        return "rejected"
    if status == "input_error":
        return "unavailable"
    return "-"
