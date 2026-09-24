from __future__ import annotations

import sqlite3
from pathlib import Path

from pump_fault_app.domain.formal_contract import FORMAL_MODEL_BUNDLE_PATH, FORMAL_MODEL_VERSION
from pump_fault_app.model_registry.models import ModelVersionRecord


class SQLiteModelRegistry:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def current_formal_version(self) -> ModelVersionRecord:
        return ModelVersionRecord(
            version=FORMAL_MODEL_VERSION,
            bundle_path=FORMAL_MODEL_BUNDLE_PATH,
            trained_at="-",
            sample_count=0,
            accuracy=None,
            macro_f1=None,
            status="当前正式模型（冻结）",
        )

    def add_candidate(
        self, *, version: str, bundle_path: Path, trained_at: str, sample_count: int,
        accuracy: float, macro_f1: float,
    ) -> ModelVersionRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO candidate_model_versions (
                    version, bundle_path, trained_at, sample_count, accuracy, macro_f1, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (version, str(bundle_path.resolve()), trained_at, sample_count, accuracy, macro_f1, "待人工批准"),
            )
        return self.get(version)

    def get(self, version: str) -> ModelVersionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM candidate_model_versions WHERE version = ?", (version,)
            ).fetchone()
        if row is None:
            raise KeyError(version)
        return self._from_row(row)

    def list_candidates(self) -> list[ModelVersionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM candidate_model_versions ORDER BY trained_at DESC, id DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def approve(self, version: str) -> ModelVersionRecord:
        with self._connect() as connection:
            connection.execute(
                "UPDATE candidate_model_versions SET status = ? WHERE version = ?",
                ("已批准（未接入正式推理）", version),
            )
        return self.get(version)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL UNIQUE,
                    bundle_path TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    macro_f1 REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ModelVersionRecord:
        return ModelVersionRecord(
            version=str(row["version"]),
            bundle_path=Path(str(row["bundle_path"])),
            trained_at=str(row["trained_at"]),
            sample_count=int(row["sample_count"]),
            accuracy=float(row["accuracy"]),
            macro_f1=float(row["macro_f1"]),
            status=str(row["status"]),
        )
