from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from pump_diagnosis.inference_contract import (
    FORMAL_BUNDLE_KEYS,
    FORMAL_FEATURE_NAMES,
    FORMAL_LABEL_ORDER,
    FORMAL_MODEL_BUNDLE_PATH,
    FORMAL_MODEL_VERSION,
    FORMAL_SOFTWARE_VERSION,
    FORMAL_V2_CONTRACT,
    validate_feature_names,
    validate_label_order,
    validate_model_bundle,
)


class _ProbabilityModel:
    def __init__(self, classes_: tuple[str, ...]) -> None:
        self.classes_ = classes_

    def predict_proba(self, values):
        return values


def test_formal_contract_uses_v2_parameters() -> None:
    contract = FORMAL_V2_CONTRACT

    assert contract.signal.target_sampling_rate == 12_000
    assert contract.signal.filter_low_hz == 10.0
    assert contract.signal.filter_high_hz == 5000.0
    assert contract.signal.window_size == 2400
    assert contract.signal.step_size == 1200
    assert contract.signal.wavelet == "db6"
    assert contract.signal.wavelet_level == 3
    assert contract.signal.envelope_low_hz == 2000.0
    assert contract.signal.envelope_high_hz == 5000.0
    assert contract.signal.harmonic_search_half_width_hz == 5.0
    assert contract.software.software_version == FORMAL_SOFTWARE_VERSION
    assert contract.model.model_version == FORMAL_MODEL_VERSION
    assert contract.model.bundle_path == FORMAL_MODEL_BUNDLE_PATH


def test_formal_feature_order_is_frozen() -> None:
    assert FORMAL_FEATURE_NAMES == (
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
    assert FORMAL_V2_CONTRACT.features.names == FORMAL_FEATURE_NAMES


def test_formal_label_order_is_frozen() -> None:
    assert FORMAL_LABEL_ORDER == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    )
    assert FORMAL_V2_CONTRACT.labels.names == FORMAL_LABEL_ORDER


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


def test_missing_bundle_fields_raise(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bad_bundle.joblib"
    joblib.dump({"model": _ProbabilityModel(FORMAL_LABEL_ORDER)}, bundle_path)

    with pytest.raises(ValueError, match="missing required keys"):
        validate_model_bundle(bundle_path)


def test_window_and_filter_limits_pass_validation() -> None:
    FORMAL_V2_CONTRACT.validate()
    assert FORMAL_V2_CONTRACT.signal.filter_high_hz < FORMAL_V2_CONTRACT.signal.target_sampling_rate / 2.0
    assert FORMAL_V2_CONTRACT.signal.window_size == 2400
    assert FORMAL_V2_CONTRACT.signal.step_size == 1200


def test_bundle_validation_accepts_formal_feature_order(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    bundle = {
        "imputer": object(),
        "scaler": object(),
        "model": _ProbabilityModel(FORMAL_LABEL_ORDER),
        "features": list(FORMAL_FEATURE_NAMES),
    }
    joblib.dump(bundle, bundle_path)

    loaded = validate_model_bundle(bundle_path)

    assert tuple(loaded["features"]) == FORMAL_FEATURE_NAMES
    assert set(loaded) == FORMAL_BUNDLE_KEYS


def test_default_formal_bundle_path_points_to_v2_bp_bundle() -> None:
    assert FORMAL_MODEL_BUNDLE_PATH.as_posix().endswith(
        "实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib"
    )


def test_formal_contract_sections_are_the_only_public_parameter_groups() -> None:
    contract = FORMAL_V2_CONTRACT

    assert contract.signal.target_sampling_rate == 12_000
    assert contract.features.names == FORMAL_FEATURE_NAMES
    assert contract.labels.names == FORMAL_LABEL_ORDER
    assert contract.model.bundle_path == FORMAL_MODEL_BUNDLE_PATH
    assert contract.software.software_version
