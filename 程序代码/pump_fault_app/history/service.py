from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pump_fault_app.history.models import DiagnosisHistoryRecord
from pump_fault_app.history.repository import SQLiteDiagnosisHistoryRepository
from pump_fault_app.sample_repository.service import DEFAULT_SAMPLE_DATABASE_PATH
from pump_fault_app.services.report_export_service import export_single_diagnosis_report


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DATABASE_PATH = PROGRAM_ROOT / "runtime_data" / "pump_fault_history.sqlite3"
DEFAULT_HISTORY_REPORT_DIR = PROGRAM_ROOT / "runtime_reports"


def calculate_window_consistency(result: Any) -> float:
    label = result.summary.diagnosis_label
    predictions = result.inference_result.window_predictions
    if label is None or not predictions:
        return 0.0
    matching_count = sum(1 for item in predictions if item.predicted_label == label)
    return matching_count / len(predictions)


def record_single_diagnosis_history(
    result: Any,
    *,
    database_path: str | Path = DEFAULT_HISTORY_DATABASE_PATH,
    report_dir: str | Path = DEFAULT_HISTORY_REPORT_DIR,
    sample_database_path: str | Path = DEFAULT_SAMPLE_DATABASE_PATH,
    diagnosed_at: datetime | None = None,
) -> DiagnosisHistoryRecord:
    if not result.summary.success:
        raise ValueError("only successful diagnoses can be stored in history")

    timestamp = diagnosed_at or datetime.now()
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / _build_report_file_name(
        file_name=str(result.summary.file_name or "diagnosis"),
        timestamp=timestamp,
    )
    export_single_diagnosis_report(result, report_path, format="docx")

    record = DiagnosisHistoryRecord(
        diagnosed_at=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        file_name=str(result.summary.file_name or "-"),
        sampling_rate_hz=int(result.summary.sampling_rate_hz),
        rpm=float(result.summary.rpm),
        predicted_label=str(result.summary.diagnosis_label),
        confidence=float(result.summary.confidence),
        window_consistency=calculate_window_consistency(result),
        report_path=report_path.resolve(),
    )
    saved_record = SQLiteDiagnosisHistoryRepository(database_path).add(record)
    from pump_fault_app.sample_repository import SQLiteSampleRepository

    predictions = result.inference_result.window_predictions or ()
    SQLiteSampleRepository(sample_database_path).store_feature_snapshots(
        history_record_id=int(saved_record.id),
        feature_vectors=tuple(item.feature_vector for item in predictions),
    )
    return saved_record


def _build_report_file_name(*, file_name: str, timestamp: datetime) -> str:
    source_stem = Path(file_name).stem
    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", source_stem).strip("_")
    if not safe_stem:
        safe_stem = "diagnosis"
    time_text = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{time_text}_{safe_stem}_{uuid4().hex[:8]}.docx"
