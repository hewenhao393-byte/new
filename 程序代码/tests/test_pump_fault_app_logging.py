from __future__ import annotations

from pathlib import Path

from pump_fault_app.app.bootstrap import bootstrap_application


def test_bootstrap_initializes_file_logging_and_writes_message(tmp_path: Path) -> None:
    state = bootstrap_application(project_root=tmp_path)

    log_file = Path(state.summary["log_file"])
    assert log_file.exists()

    contents = log_file.read_text(encoding="utf-8")

    assert "pump_fault_app bootstrap complete" in contents
    assert state.context.run_id in contents
    assert "bootstrap" in contents
