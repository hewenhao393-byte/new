from __future__ import annotations

from dataclasses import dataclass

from pump_fault_app.config.defaults import FINAL_FEATURE_NAMES


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    feature_names: tuple[str, ...]


FEATURE_GROUPS = (
    FeatureGroup(
        name="time_and_rotation",
        feature_names=(
            "kurtosis",
            "skewness",
            "crest_factor",
            "impulse_factor",
            "clearance_factor",
            "shape_factor",
            "rot_2x_1x_ratio",
            "rot_3x_1x_ratio",
            "harmonic_energy_ratio_1x_5x",
        ),
    ),
    FeatureGroup(
        name="frequency",
        feature_names=(
            "spectral_entropy",
            "spectral_flatness",
        ),
    ),
    FeatureGroup(
        name="wavelet_packet",
        feature_names=(
            "wp_energy_ratio_0",
            "wp_energy_ratio_1",
            "wp_energy_ratio_2",
            "wp_energy_ratio_3",
            "wp_energy_ratio_4",
            "wp_energy_ratio_5",
            "wp_energy_ratio_6",
            "wp_energy_ratio_7",
        ),
    ),
    FeatureGroup(
        name="envelope",
        feature_names=(
            "env_kurtosis",
            "env_crest_factor",
        ),
    ),
)


def feature_name_tuple() -> tuple[str, ...]:
    return FINAL_FEATURE_NAMES
