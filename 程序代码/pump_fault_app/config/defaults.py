from __future__ import annotations

from pathlib import Path


SIX_CLASS_LABELS = (
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
)

FINAL_FEATURE_NAMES = (
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

TARGET_SAMPLE_RATE_HZ = 12_000
BANDPASS_LOW_HZ = 10.0
BANDPASS_HIGH_HZ = 5000.0
WINDOW_SIZE = 2400
WINDOW_STEP = 1200
ENVELOPE_BAND_LOW_HZ = 2000.0
ENVELOPE_BAND_HIGH_HZ = 5000.0
WAVELET_NAME = "db6"
WAVELET_LEVEL = 3


def default_runtime_root(project_root: Path) -> Path:
    return project_root / "runtime_logs"
