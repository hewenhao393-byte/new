from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest
from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_V3_CONTRACT
from pump_fault_app.inference import run_multichannel_inference
from pump_fault_app.prediction.catboost_loader import LoadedCatBoost43Models


class _FakeModel:
    def __init__(self) -> None:
        self.classes_ = np.array(["松动", "正常", "汽蚀", "联轴器不对中", "转子不平衡", "轴承故障"], dtype=object)

    def predict_proba(self, values):
        return np.tile(np.array([[0.10, 0.20, 0.15, 0.05, 0.40, 0.10]]), (len(values), 1))


def _write_signal_csv(path: Path, samples: np.ndarray) -> None:
    pd.DataFrame({"signal": samples}).to_csv(path, index=False)


def _loaded_models() -> LoadedCatBoost43Models:
    models = MappingProxyType({channel: _FakeModel() for channel in ("CH3", "CH4", "CH5")})
    return LoadedCatBoost43Models(models, FORMAL_FEATURE_NAMES, (), MappingProxyType({}), Path("."))


def test_run_multichannel_inference_returns_probability_fusion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("pump_fault_app.inference.multichannel_inference.load_catboost43_models", lambda _: _loaded_models())

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
        + 0.05 * np.random.default_rng(7).normal(size=4800)
    )
    ch3, ch4 = tmp_path / "ch3.csv", tmp_path / "ch4.csv"
    _write_signal_csv(ch3, samples)
    _write_signal_csv(ch4, samples * 1.1)
    request = MultiChannelInferenceRequest(
        (ChannelInput("CH3", ch3, "signal"), ChannelInput("CH4", ch4, "signal")), 12_000, 1500.0
    )
    result = run_multichannel_inference(request)
    assert result.status == "diagnosed"
    assert result.valid_channels == ("CH3", "CH4")
    assert result.predicted_label == "转子不平衡"
    assert result.fused_probabilities == (0.20, 0.40, 0.05, 0.10, 0.10, 0.15)
    assert all(item.window_count == 1 for item in result.channel_results)
    for item in result.channel_results:
        assert item.visualization is not None
        assert item.visualization.time_domain is not None
        assert item.visualization.frequency_spectrum is not None
        assert item.visualization.envelope_spectrum is not None
        assert item.visualization.wavelet_packet_energy is not None


def test_v3_envelope_band_matches_accepted_feature_contract() -> None:
    assert FORMAL_V3_CONTRACT.envelope_low_hz == 1000.0
    assert FORMAL_V3_CONTRACT.envelope_high_hz == 5000.0


def test_invalid_channel_is_excluded_and_all_invalid_returns_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("pump_fault_app.inference.multichannel_inference.load_catboost43_models", lambda _: _loaded_models())
    index = np.arange(4800)
    valid = np.sin(2 * np.pi * 25 * index / 12_000.0)
    zero = np.zeros(4800)
    paths = {name: tmp_path / f"{name}.csv" for name in ("ch3", "ch4", "ch5")}
    _write_signal_csv(paths["ch3"], zero)
    _write_signal_csv(paths["ch4"], valid)
    _write_signal_csv(paths["ch5"], zero)

    partial = run_multichannel_inference(
        MultiChannelInferenceRequest(
            (ChannelInput("CH3", paths["ch3"], "signal"), ChannelInput("CH4", paths["ch4"], "signal")),
            12_000,
            1500.0,
        )
    )
    assert partial.status == "diagnosed"
    assert partial.valid_channels == ("CH4",)
    assert partial.invalid_channels == ("CH3",)

    failed = run_multichannel_inference(
        MultiChannelInferenceRequest(
            (ChannelInput("CH3", paths["ch3"], "signal"), ChannelInput("CH5", paths["ch5"], "signal")),
            12_000,
            1500.0,
        )
    )
    assert failed.status == "failed"
    assert failed.valid_channels == ()
    assert failed.fused_probabilities is None
