from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pump_fault_app.domain.diagnosis_models import ChannelInput
from pump_fault_app.services import run_single_diagnosis
from pump_fault_app.ui.pages.v3_pages import build_multichannel_run_request


def _write_signal(path: Path, *, scale: float = 1.0, invalid: bool = False) -> None:
    index = np.arange(4_800, dtype=np.float64)
    if invalid:
        samples = np.zeros_like(index)
    else:
        samples = scale * (
            0.8 * np.sin(2 * np.pi * 25 * index / 12_000)
            + 0.2 * np.sin(2 * np.pi * 50 * index / 12_000)
            + 0.03 * np.random.default_rng(2026).normal(size=index.size)
        )
    pd.DataFrame({"signal": samples}).to_csv(path, index=False)


@pytest.mark.parametrize("channels", [("CH3",), ("CH4",), ("CH3", "CH4"), ("CH3", "CH4", "CH5")])
def test_v3_end_to_end_channel_combinations(channels: tuple[str, ...], tmp_path: Path) -> None:
    inputs = []
    for offset, channel in enumerate(channels):
        path = tmp_path / f"{channel}.csv"
        _write_signal(path, scale=1.0 + 0.1 * offset)
        inputs.append(ChannelInput(channel, path, "signal"))
    request = build_multichannel_run_request(
        channels=tuple(inputs),
        sampling_rate_hz=12_000,
        rpm=1500.0,
        history_database_path=tmp_path / "history_v3.sqlite3",
        history_report_dir=tmp_path / "reports",
    )
    app_result = run_single_diagnosis(request)
    assert app_result.inference_result.status == "diagnosed"
    assert app_result.inference_result.valid_channels == channels
    assert sum(app_result.inference_result.fused_probabilities or ()) == pytest.approx(1.0)
    assert app_result.history_record.contract_version == "formal-V3-catboost43"
    assert app_result.report_path.is_file()


def test_v3_end_to_end_excludes_invalid_channel_and_fails_when_all_invalid(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.csv"
    invalid_path = tmp_path / "invalid.csv"
    _write_signal(valid_path)
    _write_signal(invalid_path, invalid=True)
    partial = run_single_diagnosis(
        build_multichannel_run_request(
            channels=(ChannelInput("CH3", invalid_path, "signal"), ChannelInput("CH4", valid_path, "signal")),
            sampling_rate_hz=12_000,
            rpm=1500.0,
            history_database_path=tmp_path / "partial.sqlite3",
            history_report_dir=tmp_path / "partial_reports",
        )
    )
    assert partial.inference_result.status == "diagnosed"
    assert partial.inference_result.valid_channels == ("CH4",)
    assert partial.inference_result.invalid_channels == ("CH3",)

    failed = run_single_diagnosis(
        build_multichannel_run_request(
            channels=(ChannelInput("CH3", invalid_path, "signal"),),
            sampling_rate_hz=12_000,
            rpm=1500.0,
            history_database_path=tmp_path / "failed.sqlite3",
            history_report_dir=tmp_path / "failed_reports",
        )
    )
    assert failed.inference_result.status == "failed"
    assert failed.history_record is None
