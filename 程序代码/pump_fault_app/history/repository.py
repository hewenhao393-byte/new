from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from pump_fault_app.history.models import DiagnosisHistoryRecord


class SQLiteDiagnosisHistoryRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add(self, record: DiagnosisHistoryRecord) -> DiagnosisHistoryRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnosis_history (
                    diagnosed_at,
                    file_name,
                    sampling_rate_hz,
                    rpm,
                    predicted_label,
                    confidence,
                    window_consistency,
                    report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.diagnosed_at,
                    record.file_name,
                    record.sampling_rate_hz,
                    record.rpm,
                    record.predicted_label,
                    record.confidence,
                    record.window_consistency,
                    str(record.report_path.resolve()),
                ),
            )
            record_id = int(cursor.lastrowid)
        return replace(record, id=record_id, report_path=record.report_path.resolve())

    def get(self, record_id: int) -> DiagnosisHistoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM diagnosis_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        return None if row is None else self._record_from_row(row)

    def list_all(self) -> list[DiagnosisHistoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM diagnosis_history ORDER BY diagnosed_at DESC, id DESC"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete(self, record_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM diagnosis_history WHERE id = ?",
                (record_id,),
            )
        return cursor.rowcount == 1

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    diagnosed_at TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    sampling_rate_hz INTEGER NOT NULL,
                    rpm REAL NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    window_consistency REAL NOT NULL,
                    report_path TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> DiagnosisHistoryRecord:
        return DiagnosisHistoryRecord(
            id=int(row["id"]),
            diagnosed_at=str(row["diagnosed_at"]),
            file_name=str(row["file_name"]),
            sampling_rate_hz=int(row["sampling_rate_hz"]),
            rpm=float(row["rpm"]),
            predicted_label=str(row["predicted_label"]),
            confidence=float(row["confidence"]),
            window_consistency=float(row["window_consistency"]),
            report_path=Path(str(row["report_path"])),
        )
