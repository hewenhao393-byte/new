from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult
from pump_fault_app.history.v3_models import V3HistoryRecord
from pump_fault_app.history.v3_repository import V3HistoryRepository


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DATABASE_PATH = PROGRAM_ROOT / "runtime_data" / "pump_fault_history_v3.sqlite3"
DEFAULT_HISTORY_REPORT_DIR = PROGRAM_ROOT / "runtime_reports"


def record_single_diagnosis_history(
    result: MultiChannelDiagnosisResult,
    *,
    source_files: Mapping[str, str],
    sampling_rate_hz: int,
    rpm: float,
    database_path: str | Path = DEFAULT_HISTORY_DATABASE_PATH,
    report_path: Path | None = None,
    diagnosed_at: datetime | None = None,
) -> V3HistoryRecord:
    if result.status != "diagnosed":
        raise ValueError("only successful diagnoses can be stored in history")
    return V3HistoryRepository(database_path).save(
        result,
        source_files=source_files,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        report_path=report_path,
        diagnosed_at=diagnosed_at,
    )
