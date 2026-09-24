from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pump_fault_app.history.models import DiagnosisHistoryRecord
from pump_fault_app.history.repository import SQLiteDiagnosisHistoryRepository


HistoryDeletionStatus = Literal[
    "deleted",
    "record_deleted_report_missing",
    "record_not_found",
    "report_delete_failed",
    "report_path_outside_system_directory",
    "report_deleted_record_delete_failed",
]


@dataclass(frozen=True)
class HistoryDeletionResult:
    status: HistoryDeletionStatus


def delete_history_record(
    repository: SQLiteDiagnosisHistoryRepository,
    record: DiagnosisHistoryRecord,
    *,
    report_root: str | Path,
    remove_file: Callable[[Path], None] = Path.unlink,
) -> HistoryDeletionResult:
    if record.id is None or repository.get(record.id) is None:
        return HistoryDeletionResult("record_not_found")

    report_path = record.report_path.resolve()
    allowed_root = Path(report_root).resolve()
    if not report_path.is_relative_to(allowed_root):
        return HistoryDeletionResult("report_path_outside_system_directory")

    if report_path.is_file():
        try:
            remove_file(report_path)
        except OSError:
            return HistoryDeletionResult("report_delete_failed")
        if not repository.delete(record.id):
            return HistoryDeletionResult("report_deleted_record_delete_failed")
        return HistoryDeletionResult("deleted")

    if repository.delete(record.id):
        return HistoryDeletionResult("record_deleted_report_missing")
    return HistoryDeletionResult("record_not_found")
