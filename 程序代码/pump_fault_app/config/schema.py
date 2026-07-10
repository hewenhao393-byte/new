from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SignalConfig:
    target_sample_rate_hz: int
    bandpass_low_hz: float
    bandpass_high_hz: float
    window_size: int
    window_step: int
    envelope_band_low_hz: float
    envelope_band_high_hz: float
    wavelet_name: str
    wavelet_level: int


@dataclass(frozen=True)
class LabelConfig:
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureConfig:
    final_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    runtime_root: Path


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str
    log_level: str
    log_filename: str


@dataclass(frozen=True)
class AppConfig:
    signal: SignalConfig
    labels: LabelConfig
    features: FeatureConfig
    paths: PathConfig
    logging: LoggingConfig

    def as_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))
