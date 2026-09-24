from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES
from pump_fault_app.domain.labels import assert_valid_label
from pump_fault_app.domain.records import FeatureVector
from pump_fault_app.sample_repository.models import ConfirmedAnnotation


class SQLiteSampleRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def store_feature_snapshots(
        self,
        *,
        history_record_id: int,
        feature_vectors: tuple[FeatureVector, ...],
    ) -> int:
        for feature_vector in feature_vectors:
            if feature_vector.feature_names != FORMAL_FEATURE_NAMES:
                raise ValueError("feature vector does not match frozen 21-feature order")
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM feature_snapshots WHERE history_record_id = ?",
                (history_record_id,),
            )
            connection.executemany(
                """
                INSERT INTO feature_snapshots (
                    history_record_id, window_index, feature_values_json
                ) VALUES (?, ?, ?)
                """,
                [
                    (
                        history_record_id,
                        window_index,
                        json.dumps(list(feature_vector.values)),
                    )
                    for window_index, feature_vector in enumerate(feature_vectors)
                ],
            )
        return len(feature_vectors)

    def save_annotation(
        self,
        *,
        history_record_id: int,
        true_label: str,
        confirmed_at: datetime | None = None,
    ) -> ConfirmedAnnotation:
        assert_valid_label(true_label)
        timestamp = confirmed_at or datetime.now()
        value = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sample_annotations (
                    history_record_id, true_label, confirmed_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(history_record_id) DO UPDATE SET
                    true_label = excluded.true_label,
                    confirmed_at = excluded.confirmed_at
                """,
                (history_record_id, true_label, value),
            )
        return ConfirmedAnnotation(
            history_record_id=history_record_id,
            true_label=true_label,
            confirmed_at=value,
        )

    def get_annotation(self, history_record_id: int) -> ConfirmedAnnotation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sample_annotations WHERE history_record_id = ?",
                (history_record_id,),
            ).fetchone()
        if row is None:
            return None
        return ConfirmedAnnotation(
            history_record_id=int(row["history_record_id"]),
            true_label=str(row["true_label"]),
            confirmed_at=str(row["confirmed_at"]),
        )

    def snapshot_count(self, *, history_record_id: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM feature_snapshots WHERE history_record_id = ?",
                (history_record_id,),
            ).fetchone()
        return int(row["count"])

    def list_training_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    snapshot.history_record_id,
                    snapshot.window_index,
                    snapshot.feature_values_json,
                    annotation.true_label
                FROM feature_snapshots AS snapshot
                INNER JOIN sample_annotations AS annotation
                    ON annotation.history_record_id = snapshot.history_record_id
                ORDER BY snapshot.history_record_id, snapshot.window_index
                """
            ).fetchall()
        return [
            {
                "history_record_id": int(row["history_record_id"]),
                "window_index": int(row["window_index"]),
                "true_label": str(row["true_label"]),
                **dict(zip(FORMAL_FEATURE_NAMES, json.loads(str(row["feature_values_json"])))),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_record_id INTEGER NOT NULL,
                    window_index INTEGER NOT NULL,
                    feature_values_json TEXT NOT NULL,
                    UNIQUE(history_record_id, window_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sample_annotations (
                    history_record_id INTEGER PRIMARY KEY,
                    true_label TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL
                )
                """
            )
