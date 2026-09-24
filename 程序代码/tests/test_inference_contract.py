from __future__ import annotations

import pytest

from pump_fault_app.domain.formal_contract import (
    FORMAL_FEATURE_NAMES,
    FORMAL_LABEL_ORDER,
    FORMAL_V3_CONTRACT,
    validate_feature_names,
    validate_label_order,
)
from pump_fault_app.domain.labels import display_label
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


def test_v3_contract_is_the_only_deployed_contract() -> None:
    contract = FORMAL_V3_CONTRACT
    assert APP_VERSION == "pump-fault-app-v3"
    assert MODEL_VERSION == "catboost43-six-class-v3"
    assert FEATURE_VERSION == "43-feature-v1"
    assert INFERENCE_CONTRACT_VERSION == "formal-V3-catboost43"
    assert contract.signal.target_sampling_rate == 12_000
    assert contract.signal.filter_low_hz == 5.0
    assert contract.signal.filter_high_hz == 5000.0
    assert contract.signal.window_size == 4800
    assert contract.signal.step_size == 2400
    assert contract.signal.wavelet == "db6"
    assert contract.signal.wavelet_level == 3
    assert contract.channels == ("CH3", "CH4", "CH5")
    assert display_label("松动") == "机械松动"


def test_formal_feature_order_is_frozen() -> None:
    assert FORMAL_FEATURE_NAMES == (
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
    assert FORMAL_V3_CONTRACT.features.names == FORMAL_FEATURE_NAMES


def test_formal_label_order_is_frozen() -> None:
    assert FORMAL_LABEL_ORDER == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    )
    assert FORMAL_V3_CONTRACT.labels.names == FORMAL_LABEL_ORDER


def test_invalid_feature_order_raises() -> None:
    bad = list(FORMAL_FEATURE_NAMES)
    bad[0], bad[1] = bad[1], bad[0]

    with pytest.raises(ValueError, match="formal feature names"):
        validate_feature_names(tuple(bad))


def test_invalid_label_order_raises() -> None:
    bad = list(FORMAL_LABEL_ORDER)
    bad[0], bad[1] = bad[1], bad[0]

    with pytest.raises(ValueError, match="formal label order"):
        validate_label_order(tuple(bad))


def test_window_and_filter_limits_pass_validation() -> None:
    FORMAL_V3_CONTRACT.validate()
    assert FORMAL_V3_CONTRACT.signal.filter_high_hz < FORMAL_V3_CONTRACT.signal.target_sampling_rate / 2.0
    assert FORMAL_V3_CONTRACT.signal.window_size == 4800
    assert FORMAL_V3_CONTRACT.signal.step_size == 2400


def test_formal_contract_sections_are_the_only_public_parameter_groups() -> None:
    contract = FORMAL_V3_CONTRACT

    assert contract.signal.target_sampling_rate == 12_000
    assert contract.features.names == FORMAL_FEATURE_NAMES
    assert contract.labels.names == FORMAL_LABEL_ORDER
    assert contract.channels == ("CH3", "CH4", "CH5")
