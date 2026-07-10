"""Frozen V2 inference contract owned by the diagnosis application.

The legacy ``pump_diagnosis.inference_contract`` module re-exports this API so
experiment scripts remain compatible, while deployed application modules no
longer depend on the experiment package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from pump_fault_app.version import APP_VERSION, MODEL_VERSION


FORMAL_SOFTWARE_VERSION = APP_VERSION
FORMAL_MODEL_VERSION = MODEL_VERSION

FORMAL_LABEL_ORDER = (
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
)

FORMAL_FEATURE_NAMES = (
    "kurtosis",
    "skewness",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "spectral_entropy",
    "spectral_flatness",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "env_kurtosis",
    "env_crest_factor",
)

FORMAL_BUNDLE_KEYS = {"imputer", "scaler", "model", "features"}
FORMAL_MODEL_BUNDLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "实验结果"
    / "多转速统一六分类实验V2"
    / "six_class_models"
    / "bp"
    / "bp_bundle.joblib"
)


def validate_feature_names(feature_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(feature_names)
    if len(normalized) != 21:
        raise ValueError(f"formal feature names must contain 21 entries, got {len(normalized)}")
    if normalized != FORMAL_FEATURE_NAMES:
        raise ValueError("formal feature names do not match the frozen V2 order")
    return normalized


def validate_label_order(label_order: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(label_order)
    if len(normalized) != 6:
        raise ValueError(f"formal label order must contain 6 entries, got {len(normalized)}")
    if normalized != FORMAL_LABEL_ORDER:
        raise ValueError("formal label order does not match the frozen V2 order")
    return normalized


def _validate_model_classes(classes_: Any) -> tuple[str, ...]:
    if classes_ is None:
        raise ValueError("model is missing classes_")
    normalized = tuple(str(label) for label in classes_)
    if len(normalized) != 6 or set(normalized) != set(FORMAL_LABEL_ORDER):
        raise ValueError("model classes_ cannot be mapped to the formal label order")
    return normalized


@dataclass(frozen=True)
class SignalContract:
    target_sampling_rate: int = 12_000
    filter_low_hz: float = 10.0
    filter_high_hz: float = 5000.0
    window_size: int = 2400
    step_size: int = 1200
    wavelet: str = "db6"
    wavelet_level: int = 3
    envelope_low_hz: float = 2000.0
    envelope_high_hz: float = 5000.0
    harmonic_search_half_width_hz: float = 5.0

    def validate(self) -> "SignalContract":
        nyquist_hz = self.target_sampling_rate / 2.0
        if self.filter_low_hz <= 0.0 or self.filter_low_hz >= self.filter_high_hz:
            raise ValueError("filter band must satisfy 0 < low < high")
        if self.filter_high_hz >= nyquist_hz:
            raise ValueError("filter high cutoff must be lower than the Nyquist frequency")
        if self.envelope_low_hz <= 0.0 or self.envelope_low_hz >= self.envelope_high_hz:
            raise ValueError("envelope band must satisfy 0 < low < high")
        if self.envelope_high_hz >= nyquist_hz:
            raise ValueError("envelope high cutoff must be lower than the Nyquist frequency")
        if self.window_size <= 0 or self.step_size <= 0:
            raise ValueError("window size and step size must be positive")
        if self.step_size > self.window_size:
            raise ValueError("step size cannot exceed window size")
        if self.wavelet_level <= 0:
            raise ValueError("wavelet level must be positive")
        return self


@dataclass(frozen=True)
class FeatureContract:
    names: tuple[str, ...] = FORMAL_FEATURE_NAMES

    def validate(self) -> "FeatureContract":
        validate_feature_names(self.names)
        return self


@dataclass(frozen=True)
class LabelContract:
    names: tuple[str, ...] = FORMAL_LABEL_ORDER

    def validate(self) -> "LabelContract":
        validate_label_order(self.names)
        return self


@dataclass(frozen=True)
class ModelContract:
    bundle_path: Path = FORMAL_MODEL_BUNDLE_PATH
    model_version: str = FORMAL_MODEL_VERSION
    required_bundle_keys: frozenset[str] = frozenset(FORMAL_BUNDLE_KEYS)


@dataclass(frozen=True)
class SoftwareContract:
    software_version: str = FORMAL_SOFTWARE_VERSION


@dataclass(frozen=True)
class FormalInferenceContract:
    signal: SignalContract = SignalContract()
    features: FeatureContract = FeatureContract()
    labels: LabelContract = LabelContract()
    model: ModelContract = ModelContract()
    software: SoftwareContract = SoftwareContract()

    @property
    def target_sampling_rate(self) -> int:
        return self.signal.target_sampling_rate

    @property
    def filter_low_hz(self) -> float:
        return self.signal.filter_low_hz

    @property
    def filter_high_hz(self) -> float:
        return self.signal.filter_high_hz

    @property
    def window_size(self) -> int:
        return self.signal.window_size

    @property
    def step_size(self) -> int:
        return self.signal.step_size

    @property
    def wavelet(self) -> str:
        return self.signal.wavelet

    @property
    def wavelet_level(self) -> int:
        return self.signal.wavelet_level

    @property
    def envelope_low_hz(self) -> float:
        return self.signal.envelope_low_hz

    @property
    def envelope_high_hz(self) -> float:
        return self.signal.envelope_high_hz

    @property
    def harmonic_search_half_width_hz(self) -> float:
        return self.signal.harmonic_search_half_width_hz

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.features.names

    @property
    def label_order(self) -> tuple[str, ...]:
        return self.labels.names

    @property
    def model_bundle_path(self) -> Path:
        return self.model.bundle_path

    def validate(self) -> "FormalInferenceContract":
        self.signal.validate()
        self.features.validate()
        self.labels.validate()
        if not self.software.software_version:
            raise ValueError("software version must be non-empty")
        if not self.model.model_version:
            raise ValueError("model version must be non-empty")
        return self


FORMAL_V2_CONTRACT = FormalInferenceContract().validate()


def validate_model_bundle(bundle_path: Path | str | None = None) -> dict[str, Any]:
    path = FORMAL_MODEL_BUNDLE_PATH if bundle_path is None else Path(bundle_path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a dict-like object")
    missing = FORMAL_BUNDLE_KEYS.difference(bundle.keys())
    if missing:
        raise ValueError(f"model bundle is missing required keys: {sorted(missing)}")
    features = validate_feature_names(tuple(bundle["features"]))
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("model bundle model must support predict_proba")
    _validate_model_classes(getattr(model, "classes_", None))
    return {
        "imputer": bundle["imputer"],
        "scaler": bundle["scaler"],
        "model": model,
        "features": features,
    }
