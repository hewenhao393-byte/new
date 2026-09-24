from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

from pump_fault_app.domain.diagnosis_models import MultiChannelDiagnosisResult
from pump_fault_app.history.v3_models import V3HistoryRecord


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class V3HistoryRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def save(
        self,
        result: MultiChannelDiagnosisResult,
        *,
        source_files: Mapping[str, str],
        sampling_rate_hz: int,
        rpm: float,
        report_path: Path | None = None,
        diagnosed_at: datetime | None = None,
    ) -> V3HistoryRecord:
        timestamp = (diagnosed_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        channel_payload = tuple(
            {
                "channel": item.channel,
                "status": item.status,
                "failure_stage": item.failure_stage,
                "failure_message": item.failure_message,
                "predicted_label": item.predicted_label,
                "class_probabilities": list(item.class_probabilities) if item.class_probabilities else None,
                "window_count": item.window_count,
                "valid_window_count": item.valid_window_count,
                "window_consistency": item.window_consistency,
                "warnings": list(item.warnings),
            }
            for item in result.channel_results
        )
        record = V3HistoryRecord(
            diagnosed_at=timestamp,
            source_files={channel: str(source_files[channel]) for channel in result.input_channels},
            input_channels=result.input_channels,
            sampling_rate_hz=int(sampling_rate_hz),
            rpm=float(rpm),
            status=result.status,
            predicted_label=result.predicted_label,
            fused_probabilities=result.fused_probabilities,
            channel_results=channel_payload,
            agreement_level=result.agreement_level,
            agreement_count=result.agreement_count,
            model_version=result.model_version,
            feature_version=result.feature_version,
            contract_version=result.contract_version,
            warnings=result.warnings,
            report_path=report_path.resolve() if report_path is not None else None,
        )
        return self.add(record)

    def add(self, record: V3HistoryRecord) -> V3HistoryRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnosis_history_v3 (
                    diagnosed_at, source_files_json, input_channels_json,
                    sampling_rate_hz, rpm, status, predicted_label,
                    fused_probabilities_json, channel_results_json,
                    agreement_level, agreement_count, model_version,
                    feature_version, contract_version, warnings_json, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.diagnosed_at,
                    _json(record.source_files),
                    _json(record.input_channels),
                    record.sampling_rate_hz,
                    record.rpm,
                    record.status,
                    record.predicted_label,
                    _json(record.fused_probabilities),
                    _json(record.channel_results),
                    record.agreement_level,
                    record.agreement_count,
                    record.model_version,
                    record.feature_version,
                    record.contract_version,
                    _json(record.warnings),
                    str(record.report_path) if record.report_path is not None else None,
                ),
            )
            record_id = int(cursor.lastrowid)
        return replace(record, id=record_id)

    def get(self, record_id: int) -> V3HistoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM diagnosis_history_v3 WHERE id = ?", (record_id,)
            ).fetchone()
        return None if row is None else self._from_row(row)

    def list_all(self) -> list[V3HistoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM diagnosis_history_v3 ORDER BY diagnosed_at DESC, id DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, record_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM diagnosis_history_v3 WHERE id = ?", (record_id,))
        return cursor.rowcount == 1

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_history_v3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    diagnosed_at TEXT NOT NULL,
                    source_files_json TEXT NOT NULL,
                    input_channels_json TEXT NOT NULL,
                    sampling_rate_hz INTEGER NOT NULL,
                    rpm REAL NOT NULL,
                    status TEXT NOT NULL,
                    predicted_label TEXT,
                    fused_probabilities_json TEXT NOT NULL,
                    channel_results_json TEXT NOT NULL,
                    agreement_level TEXT,
                    agreement_count INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    contract_version TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    report_path TEXT
                )
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> V3HistoryRecord:
        probabilities = json.loads(row["fused_probabilities_json"])
        return V3HistoryRecord(
            id=int(row["id"]),
            diagnosed_at=str(row["diagnosed_at"]),
            source_files=dict(json.loads(row["source_files_json"])),
            input_channels=tuple(json.loads(row["input_channels_json"])),
            sampling_rate_hz=int(row["sampling_rate_hz"]),
            rpm=float(row["rpm"]),
            status=str(row["status"]),
            predicted_label=row["predicted_label"],
            fused_probabilities=None if probabilities is None else tuple(float(v) for v in probabilities),
            channel_results=tuple(dict(item) for item in json.loads(row["channel_results_json"])),
            agreement_level=row["agreement_level"],
            agreement_count=int(row["agreement_count"]),
            model_version=str(row["model_version"]),
            feature_version=str(row["feature_version"]),
            contract_version=str(row["contract_version"]),
            warnings=tuple(json.loads(row["warnings_json"])),
            report_path=Path(row["report_path"]) if row["report_path"] else None,
        )
