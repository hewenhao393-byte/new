from __future__ import annotations

from pathlib import Path

from pump_fault_app.sample_repository.repository import SQLiteSampleRepository


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DATABASE_PATH = PROGRAM_ROOT / "runtime_data" / "pump_fault_samples.sqlite3"


def load_sample_repository(
    database_path: str | Path = DEFAULT_SAMPLE_DATABASE_PATH,
) -> SQLiteSampleRepository:
    return SQLiteSampleRepository(database_path)
