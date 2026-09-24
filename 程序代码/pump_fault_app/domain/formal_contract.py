"""The single frozen CatBoost43 V3 inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


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
    "rms",
    "std",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_1x_energy_ratio",
    "rot_2x_energy_ratio",
    "rot_3x_energy_ratio",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "harmonic_energy_ratio_3x_5x",
    "rot_2x_harmonic_ratio",
    "rot_05x_1x_ratio",
    "noninteger_harmonic_energy_ratio",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_centroid",
    "spectral_bandwidth",
    "band_energy_5_300_ratio",
    "band_energy_300_1000_ratio",
    "band_energy_1000_3000_ratio",
    "band_energy_3000_5000_ratio",
    "high_low_energy_ratio",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "wp_energy_entropy",
    "env_kurtosis",
    "env_crest_factor",
    "env_spectral_entropy",
    "env_peak_energy_ratio",
    "env_peak_concentration",
    "env_peak_count",
)


def validate_feature_names(feature_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(feature_names)
    if len(normalized) != 43:
        raise ValueError(f"formal feature names must contain 43 entries, got {len(normalized)}")
    if normalized != FORMAL_FEATURE_NAMES:
        raise ValueError("formal feature names do not match the frozen V3 order")
    return normalized


def validate_label_order(label_order: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(label_order)
    if len(normalized) != 6:
        raise ValueError(f"formal label order must contain 6 entries, got {len(normalized)}")
    if normalized != FORMAL_LABEL_ORDER:
        raise ValueError("formal label order does not match the frozen V2 order")
    return normalized


@dataclass(frozen=True)
class SignalContract:
    target_sampling_rate: int = 12_000
    filter_low_hz: float = 5.0
    filter_high_hz: float = 5000.0
    window_size: int = 4800
    step_size: int = 2400
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
    model_type: str = "CatBoost"
    model_version: str = FORMAL_MODEL_VERSION


@dataclass(frozen=True)
class SoftwareContract:
    software_version: str = FORMAL_SOFTWARE_VERSION
    feature_version: str = FEATURE_VERSION
    inference_contract_version: str = INFERENCE_CONTRACT_VERSION


@dataclass(frozen=True)
class FormalInferenceContract:
    signal: SignalContract = SignalContract()
    features: FeatureContract = FeatureContract()
    labels: LabelContract = LabelContract()
    model: ModelContract = ModelContract()
    software: SoftwareContract = SoftwareContract()
    channels: tuple[str, ...] = ("CH3", "CH4", "CH5")

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

    def validate(self) -> "FormalInferenceContract":
        self.signal.validate()
        self.features.validate()
        self.labels.validate()
        if not self.software.software_version:
            raise ValueError("software version must be non-empty")
        if not self.model.model_version:
            raise ValueError("model version must be non-empty")
        if self.channels != ("CH3", "CH4", "CH5"):
            raise ValueError("formal channels must be CH3, CH4, CH5")
        return self


FORMAL_V3_CONTRACT = FormalInferenceContract().validate()
