from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.domain.labels import assert_valid_label
from pump_fault_app.version import FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


ChannelName = Literal["CH3", "CH4", "CH5"]
ChannelStatus = Literal["valid", "invalid"]
DiagnosisStatus = Literal["diagnosed", "failed"]
AgreementLevel = Literal["高一致", "中等一致", "低一致"]


def probability_tuple(values: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    array = np.asarray(normalized, dtype=np.float64)
    if array.shape != (len(FORMAL_LABEL_ORDER),):
        raise ValueError("probabilities must contain six values")
    if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    if not np.isclose(float(array.sum()), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("probabilities must sum to 1")
    return normalized


@dataclass(frozen=True)
class ChannelInput:
    channel: ChannelName
    file_path: Path
    signal_column: str | None = None
    time_column: str | None = None

    def __post_init__(self) -> None:
        if self.channel not in {"CH3", "CH4", "CH5"}:
            raise ValueError(f"unsupported channel: {self.channel}")


@dataclass(frozen=True)
class MultiChannelInferenceRequest:
    channels: tuple[ChannelInput, ...]
    sampling_rate_hz: int
    rpm: float
    model_directory: Path | None = None

    def __post_init__(self) -> None:
        names = tuple(item.channel for item in self.channels)
        if not 1 <= len(names) <= 3:
            raise ValueError("channels must contain 1 to 3 inputs")
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel input")
        if self.sampling_rate_hz <= 0 or self.rpm <= 0:
            raise ValueError("sampling rate and rpm must be positive")


@dataclass(frozen=True)
class WindowDiagnosisResult:
    window_index: int
    start_index: int
    end_index: int
    predicted_label: str
    class_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.window_index < 0 or self.end_index <= self.start_index:
            raise ValueError("invalid window indices")
        assert_valid_label(self.predicted_label)
        object.__setattr__(self, "class_probabilities", probability_tuple(self.class_probabilities))


@dataclass(frozen=True)
class ChannelDiagnosisResult:
    channel: ChannelName
    status: ChannelStatus
    failure_stage: str | None = None
    failure_message: str | None = None
    predicted_label: str | None = None
    class_probabilities: tuple[float, ...] | None = None
    window_predictions: tuple[WindowDiagnosisResult, ...] = ()
    window_count: int = 0
    valid_window_count: int = 0
    window_consistency: float | None = None
    quality_report: Any | None = None
    visualization: Any | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.channel not in {"CH3", "CH4", "CH5"}:
            raise ValueError(f"unsupported channel: {self.channel}")
        if self.status == "invalid":
            if self.predicted_label is not None or self.class_probabilities is not None:
                raise ValueError("invalid channel cannot contain a prediction")
        elif self.status == "valid":
            if self.predicted_label is None or self.class_probabilities is None:
                raise ValueError("valid channel requires a prediction")
            assert_valid_label(self.predicted_label)
            object.__setattr__(self, "class_probabilities", probability_tuple(self.class_probabilities))
        else:
            raise ValueError(f"unsupported channel status: {self.status}")
        if self.window_count != len(self.window_predictions):
            raise ValueError("window_count does not match window_predictions")
        if not 0 <= self.valid_window_count <= self.window_count:
            raise ValueError("valid_window_count is outside the window range")
        if self.window_consistency is not None and not 0.0 <= self.window_consistency <= 1.0:
            raise ValueError("window_consistency must be between 0 and 1")


@dataclass(frozen=True)
class MultiChannelDiagnosisResult:
    status: DiagnosisStatus
    input_channels: tuple[ChannelName, ...]
    valid_channels: tuple[ChannelName, ...]
    invalid_channels: tuple[ChannelName, ...]
    channel_results: tuple[ChannelDiagnosisResult, ...]
    predicted_label: str | None
    fused_probabilities: tuple[float, ...] | None
    agreement_level: AgreementLevel | None
    agreement_count: int
    valid_channel_count: int
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    contract_version: str = INFERENCE_CONTRACT_VERSION
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.valid_channel_count != len(self.valid_channels):
            raise ValueError("valid_channel_count does not match valid_channels")
        if set(self.valid_channels).intersection(self.invalid_channels):
            raise ValueError("valid and invalid channel sets overlap")
        if self.status == "failed":
            if self.predicted_label is not None or self.fused_probabilities is not None:
                raise ValueError("failed result cannot contain a final prediction")
        elif self.status == "diagnosed":
            if self.predicted_label is None or self.fused_probabilities is None:
                raise ValueError("diagnosed result requires a final prediction")
            assert_valid_label(self.predicted_label)
            object.__setattr__(self, "fused_probabilities", probability_tuple(self.fused_probabilities))
        else:
            raise ValueError(f"unsupported diagnosis status: {self.status}")
