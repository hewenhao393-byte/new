from __future__ import annotations

from pathlib import Path
import importlib

import pytest


@pytest.fixture(autouse=True)
def isolate_default_history_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    try:
        history_service = importlib.import_module("pump_fault_app.history.service")
    except ImportError:
        return
    monkeypatch.setattr(history_service, "DEFAULT_HISTORY_DATABASE_PATH", tmp_path / "default_history.sqlite3")
    monkeypatch.setattr(history_service, "DEFAULT_HISTORY_REPORT_DIR", tmp_path / "default_history_reports")
