from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pump_fault_app.history.models import DiagnosisHistoryRecord


def build_history_table_rows(
    records: Iterable[DiagnosisHistoryRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "记录ID": record.id,
            "诊断时间": record.diagnosed_at,
            "文件名": record.file_name,
            "采样率": f"{record.sampling_rate_hz} Hz",
            "转速": f"{record.rpm:.1f} rpm",
            "预测类别": _display_label(record.predicted_label),
            "置信度": f"{record.confidence:.1%}",
            "窗口一致率": f"{record.window_consistency:.1%}",
            "报告状态": _report_status(record),
        }
        for record in records
    ]


def build_history_summary(
    records: Iterable[DiagnosisHistoryRecord],
) -> dict[str, str]:
    record_list = list(records)
    return {
        "历史记录数": str(len(record_list)),
        "最近诊断时间": record_list[0].diagnosed_at if record_list else "-",
    }


def build_history_detail(record: DiagnosisHistoryRecord) -> dict[str, str]:
    return {
        "诊断时间": record.diagnosed_at,
        "文件名": record.file_name,
        "采样率": f"{record.sampling_rate_hz} Hz",
        "转速": f"{record.rpm:.1f} rpm",
        "预测类别": _display_label(record.predicted_label),
        "置信度": f"{record.confidence:.1%}",
        "窗口一致率": f"{record.window_consistency:.1%}",
        "Word报告路径": str(record.report_path),
        "报告状态": _report_status(record),
    }


def build_history_select_options(
    records: Iterable[DiagnosisHistoryRecord],
) -> dict[int, str]:
    return {
        int(record.id): (
            f"{record.diagnosed_at}｜{record.file_name}｜{_display_label(record.predicted_label)}"
        )
        for record in records
        if record.id is not None
    }


def _display_label(label: str) -> str:
    return "机械松动" if label == "松动" else label


def _report_status(record: DiagnosisHistoryRecord) -> str:
    return "可打开" if record.report_path.is_file() else "报告文件不存在"
