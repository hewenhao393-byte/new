from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    original_fs: int = 20_000
    target_fs: int = 12_000
    bandpass_low: float = 5.0
    bandpass_high: float = 5_000.0
    window_size: int = 4_800
    step_size: int = 2_400
    order_half_width: float = 2.5
    wavelet: str = "db6"
    wavelet_level: int = 3
    eps: float = 1e-12
    random_seed: int = 2026
    temporal_train_samples: int = 96_000
    temporal_guard_samples: int = 12_000
    temporal_test_samples: int = 36_000
    train_block_samples: int = 19_200

CONFIG = PipelineConfig()
FEATURE_NAMES = (
    "rms","std","peak_to_peak","skewness","kurtosis","crest_factor","impulse_factor","clearance_factor","shape_factor",
    "rot_1x_energy_ratio","rot_2x_energy_ratio","rot_3x_energy_ratio","rot_2x_1x_ratio","rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x","harmonic_energy_ratio_3x_5x","rot_2x_harmonic_ratio","rot_05x_1x_ratio","noninteger_harmonic_energy_ratio",
    "spectral_entropy","spectral_flatness","spectral_centroid","spectral_bandwidth",
    "band_energy_5_300_ratio","band_energy_300_1000_ratio","band_energy_1000_3000_ratio","band_energy_3000_5000_ratio","high_low_energy_ratio",
    *(f"wp_energy_ratio_{i}" for i in range(8)),"wp_energy_entropy","env_kurtosis","env_crest_factor","env_spectral_entropy","env_peak_energy_ratio","env_peak_concentration","env_peak_count",
)
