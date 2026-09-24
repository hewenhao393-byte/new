from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from pump_fault_app.domain.diagnosis_models import ChannelInput
from pump_fault_app.ui.pages.v3_pages import build_multichannel_run_request
from pump_fault_app.ui.streamlit_app import build_home_sections, build_navigation_items


def test_navigation_hides_model_optimization() -> None:
    labels = [item["title"] for item in build_navigation_items()]
    assert "模型持续优化" not in labels
    assert "多通道诊断" in labels


def test_home_describes_only_v3_contract() -> None:
    text = repr(build_home_sections())
    for marker in ("43维", "CatBoost", "CH3", "CH4", "CH5", "4800", "2400", "5–5000"):
        assert marker in text
    assert "BP" not in text
    assert "21维" not in text


@pytest.mark.parametrize("names", [("CH3",), ("CH3", "CH4"), ("CH3", "CH4", "CH5")])
def test_request_builder_accepts_one_to_three_separate_files(tmp_path: Path, names: tuple[str, ...]) -> None:
    channels = tuple(ChannelInput(name, tmp_path / f"{name}.csv", "signal") for name in names)
    request = build_multichannel_run_request(channels=channels, sampling_rate_hz=12_000, rpm=1500.0)
    assert request.inference_request.channels == channels


def test_request_builder_rejects_mixed_time_columns(tmp_path: Path) -> None:
    channels = (
        ChannelInput("CH3", tmp_path / "a.csv", "signal", "time"),
        ChannelInput("CH4", tmp_path / "b.csv", "signal", None),
    )
    with pytest.raises(ValueError, match="全部提供或全部不提供"):
        build_multichannel_run_request(channels=channels, sampling_rate_hz=12_000, rpm=1500.0)


def test_streamlit_entrypoint_imports_from_outside_project(tmp_path: Path) -> None:
    entrypoint = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "streamlit_app.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({str(entrypoint)!r}, run_name='entrypoint_import_test')",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
