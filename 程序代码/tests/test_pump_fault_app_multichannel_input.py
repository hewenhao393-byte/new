from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest
from pump_fault_app.inference.multichannel_inference import read_and_validate_channel_files


def _write(path: Path, length: int, *, time_shift: float = 0.0) -> None:
    index = np.arange(length, dtype=np.float64)
    pd.DataFrame(
        {
            "time": index / 12_000.0 + time_shift,
            "signal": np.sin(2 * np.pi * 25 * index / 12_000.0),
        }
    ).to_csv(path, index=False)


def _request(*items: ChannelInput) -> MultiChannelInferenceRequest:
    return MultiChannelInferenceRequest(tuple(items), sampling_rate_hz=12_000, rpm=1500.0)


def test_multichannel_rejects_mixed_time_column_presence(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, 4800)
    _write(b, 4800)
    request = _request(ChannelInput("CH3", a, "signal", "time"), ChannelInput("CH4", b, "signal", None))
    with pytest.raises(ValueError, match="all provide time columns or none"):
        read_and_validate_channel_files(request)


def test_multichannel_rejects_length_mismatch(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, 4800)
    _write(b, 4799)
    request = _request(ChannelInput("CH3", a, "signal", None), ChannelInput("CH4", b, "signal", None))
    with pytest.raises(ValueError, match="signal lengths differ"):
        read_and_validate_channel_files(request)


def test_multichannel_rejects_pointwise_time_mismatch(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, 4800)
    _write(b, 4800, time_shift=0.01)
    request = _request(ChannelInput("CH3", a, "signal", "time"), ChannelInput("CH4", b, "signal", "time"))
    with pytest.raises(ValueError, match="time axes differ"):
        read_and_validate_channel_files(request)


def test_multichannel_without_time_columns_derives_common_axis(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, 4800)
    _write(b, 4800)
    request = _request(ChannelInput("CH3", a, "signal", None), ChannelInput("CH4", b, "signal", None))
    loaded = read_and_validate_channel_files(request)
    assert tuple(loaded) == ("CH3", "CH4")
    assert np.array_equal(loaded["CH3"].time_axis, loaded["CH4"].time_axis)
    assert loaded["CH3"].time_axis[-1] == pytest.approx(4799 / 12_000)
