from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_runs_from_outside_project_and_exposes_contract_flags() -> None:
    script = Path(__file__).parents[1] / "tools" / "run_channel_fusion_weight_study.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd="/private/tmp", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for flag in ("--ch3", "--ch4", "--ch5", "--split", "--folds", "--source-manifest", "--output", "--dry-run"):
        assert flag in result.stdout
