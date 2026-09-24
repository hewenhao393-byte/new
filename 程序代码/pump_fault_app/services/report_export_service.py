from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from pump_fault_app.reporting import SingleReportViewData, build_single_report_view_data, export_single_report_to_docx

if TYPE_CHECKING:
    from pump_fault_app.services.app_service import AppSingleRunResult


SingleReportExportInput = Union["AppSingleRunResult", SingleReportViewData]


def export_single_diagnosis_report(
    result: SingleReportExportInput,
    output_path: str | Path,
    format: str = "docx",
) -> Path:
    normalized_format = format.strip().lower()
    if normalized_format != "docx":
        raise ValueError("only docx format is currently supported")

    target_path = Path(output_path)
    if target_path.suffix.lower() != ".docx":
        raise ValueError("output_path must end with .docx when exporting docx reports")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    view_data = result if isinstance(result, SingleReportViewData) else build_single_report_view_data(result)
    return export_single_report_to_docx(view_data, target_path)
