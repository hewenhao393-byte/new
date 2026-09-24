from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_default_history_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pump_fault_app.history.service.DEFAULT_HISTORY_DATABASE_PATH",
        tmp_path / "default_history.sqlite3",
    )
    monkeypatch.setattr(
        "pump_fault_app.history.service.DEFAULT_HISTORY_REPORT_DIR",
        tmp_path / "default_history_reports",
    )
