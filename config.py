from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pump_diagnosis.labels import LABEL_ORDER


DATASET_ROOT = Path("/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path = field(
        default_factory=lambda: DATASET_ROOT / "数据集" / "原始数据集" / "Vibration"
    )
    condition_workbook: Path = field(
        default_factory=lambda: DATASET_ROOT / "附录" / "其他资料" / "测量工况总表.xlsx"
    )
    output_root: Path = field(
        default_factory=lambda: Path("outputs") / "channel4_six_class_v1"
    )
    channel: int = 4
    original_fs: int = 20_000
    processed_fs: int = 12_000
    resample_up: int = 3
    resample_down: int = 5
    filter_order: int = 4
    bandpass_low: float = 10.0
    bandpass_high: float = 5000.0
    envelope_low: float = 2000.0
    envelope_high: float = 5000.0
    envelope_spectrum_high: float = 3000.0
    window_size: int = 4096
    step_size: int = 2048
    harmonic_tolerance: float = 5.0
    wavelet: str = "db4"
    wavelet_mode: str = "symmetric"
    wavelet_level: int = 3
    wavelet_ratio_tolerance: float = 1e-6
    harmonic_multiples: tuple[int, ...] = (1, 2, 3)
    envelope_harmonic_multiples: tuple[int, ...] = (1, 2)
    frequency_energy_bands: tuple[tuple[str, float, float], ...] = (
        ("energy_ratio_10_500", 10.0, 500.0),
        ("energy_ratio_500_2000", 500.0, 2000.0),
        ("energy_ratio_2000_5000", 2000.0, 5000.0),
    )
    envelope_energy_bands: tuple[tuple[str, float, float], ...] = (
        ("env_energy_ratio_0_100", 0.0, 100.0),
        ("env_energy_ratio_100_500", 100.0, 500.0),
        ("env_energy_ratio_500_1000", 500.0, 1000.0),
        ("env_energy_ratio_1000_3000", 1000.0, 3000.0),
    )
    missing_feature_threshold: float = 0.20
    correlation_threshold: float = 0.90
    near_zero_variance_threshold: float = 1e-12
    test_size: float = 0.2
    random_state: int = 42
    plot_dpi: int = 300
    validation_chunk_size: int = 25_000
    csv_encoding: str = "utf-8-sig"
    label_order: tuple[str, ...] = LABEL_ORDER

    def as_serializable_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ModelingConfig:
    feature_root: Path = field(
        default_factory=lambda: Path("outputs") / "channel4_six_class_v1"
    )
    output_root: Path = field(
        default_factory=lambda: Path("outputs") / "channel4_six_class_v1" / "modeling"
    )
    random_state: int = 42
    correlation_threshold: float = 0.90
    near_zero_variance_threshold: float = 1e-12
    top_k_candidates: tuple[int, ...] = (10, 15, 20, 25)
    cv_folds: int = 5
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    rf_class_weight: str = "balanced_subsample"
    rf_n_jobs: int = -1
    svm_sample_size: int = 16_000
    svm_kernel: str = "rbf"
    svm_c: float = 10.0
    svm_gamma: str = "scale"
    svm_class_weight: str = "balanced"
    mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32)
    mlp_activation: str = "relu"
    mlp_solver: str = "adam"
    mlp_alpha: float = 0.0001
    mlp_learning_rate_init: float = 0.001
    mlp_max_iter: int = 300
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    mlp_n_iter_no_change: int = 10
    mlp_batch_size: int | str = "auto"
    plot_dpi: int = 300
    csv_encoding: str = "utf-8-sig"
    correlation_priority: tuple[str, ...] = ("rms", "std", "variance")
    label_order: tuple[str, ...] = LABEL_ORDER

    def as_serializable_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))
