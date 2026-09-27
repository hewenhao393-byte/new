from __future__ import annotations

import json

import pytest

from fusion_weight_study.contracts import (
    CHANNELS,
    CV_FOLDS,
    FEATURE_NAMES,
    LABELS,
    MODEL_PARAMS,
    PRIMARY_SELECTION_METRIC,
    SIMPLICITY_TOLERANCE,
    ContractError,
    validate_source_manifest,
)


EXPECTED_FEATURES = (
    "rms", "std", "peak_to_peak", "skewness", "kurtosis", "crest_factor",
    "impulse_factor", "clearance_factor", "shape_factor", "rot_1x_energy_ratio",
    "rot_2x_energy_ratio", "rot_3x_energy_ratio", "rot_2x_1x_ratio",
    "rot_3x_1x_ratio", "harmonic_energy_ratio_1x_5x",
    "harmonic_energy_ratio_3x_5x", "rot_2x_harmonic_ratio",
    "rot_05x_1x_ratio", "noninteger_harmonic_energy_ratio", "spectral_entropy",
    "spectral_flatness", "spectral_centroid", "spectral_bandwidth",
    "band_energy_5_300_ratio", "band_energy_300_1000_ratio",
    "band_energy_1000_3000_ratio", "band_energy_3000_5000_ratio",
    "high_low_energy_ratio", "wp_energy_ratio_0", "wp_energy_ratio_1",
    "wp_energy_ratio_2", "wp_energy_ratio_3", "wp_energy_ratio_4",
    "wp_energy_ratio_5", "wp_energy_ratio_6", "wp_energy_ratio_7",
    "wp_energy_entropy", "env_kurtosis", "env_crest_factor",
    "env_spectral_entropy", "env_peak_energy_ratio", "env_peak_concentration",
    "env_peak_count",
)


def test_fixed_experiment_contract() -> None:
    assert CHANNELS == (3, 4, 5)
    assert LABELS == ("正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀")
    assert FEATURE_NAMES == EXPECTED_FEATURES
    assert len(FEATURE_NAMES) == 43
    assert CV_FOLDS == 3
    assert PRIMARY_SELECTION_METRIC == "record_macro_f1"
    assert SIMPLICITY_TOLERANCE == pytest.approx(0.005)
    assert MODEL_PARAMS == {
        "loss_function": "MultiClass",
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 100,
        "random_strength": 5,
        "rsm": 0.7,
        "auto_class_weights": "SqrtBalanced",
        "random_seed": 2026,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": 4,
    }


def test_manifest_must_match_fixed_contract(tmp_path) -> None:
    manifest = {
        "labels": list(LABELS),
        "features": list(FEATURE_NAMES),
        "params": dict(MODEL_PARAMS),
        "cv_splits": CV_FOLDS,
        "split_mode": "parent_group",
        "parent_identity": "record_id.rsplit('::', 1)[0]",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    validated = validate_source_manifest(path)

    assert validated["params"] == MODEL_PARAMS


def test_manifest_drift_is_a_hard_failure(tmp_path) -> None:
    manifest = {
        "labels": list(LABELS),
        "features": list(FEATURE_NAMES),
        "params": {**MODEL_PARAMS, "iterations": 499},
        "cv_splits": CV_FOLDS,
        "split_mode": "parent_group",
        "parent_identity": "record_id.rsplit('::', 1)[0]",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ContractError, match="params"):
        validate_source_manifest(path)
