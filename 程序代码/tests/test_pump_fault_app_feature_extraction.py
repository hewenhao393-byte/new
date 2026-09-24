from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES
from pump_fault_app.feature_extraction import extract_catboost43_features


def test_catboost43_feature_extraction_returns_frozen_feature_order() -> None:
    time = np.arange(4800, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 50.0 * time)
    vector = extract_catboost43_features(samples, rpm=1480.0)

    assert vector.feature_names == FORMAL_FEATURE_NAMES
    assert len(vector.values) == 43


def test_catboost43_feature_extraction_produces_finite_values() -> None:
    time = np.arange(4800, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 80.0 * time) + 0.3 * np.sin(2.0 * np.pi * 160.0 * time)
    vector = extract_catboost43_features(samples, rpm=1480.0)

    assert np.isfinite(np.asarray(vector.values, dtype=np.float64)).all()


def test_wavelet_energy_ratios_sum_to_one() -> None:
    time = np.arange(4800, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * 100.0 * time)
    vector = extract_catboost43_features(samples, rpm=1480.0)
    feature_map = dict(zip(vector.feature_names, vector.values))
    total = sum(feature_map[f"wp_energy_ratio_{index}"] for index in range(8))

    assert np.isclose(total, 1.0, atol=1e-6)


def test_rotation_ratio_features_are_non_negative() -> None:
    time = np.arange(4800, dtype=np.float64) / 12_000.0
    samples = np.sin(2.0 * np.pi * (1480.0 / 60.0) * time)
    vector = extract_catboost43_features(samples, rpm=1480.0)
    feature_map = dict(zip(vector.feature_names, vector.values))

    assert feature_map["rot_2x_1x_ratio"] >= 0.0
    assert feature_map["rot_3x_1x_ratio"] >= 0.0
    assert feature_map["harmonic_energy_ratio_1x_5x"] >= 0.0


def test_feature_extraction_rejects_wrong_window_size() -> None:
    with pytest.raises(ValueError, match="4800 finite samples"):
        extract_catboost43_features(np.ones(4799), rpm=1480.0)


def test_catboost43_matches_the_accepted_reference_implementation() -> None:
    reference_root = Path(
        "/Users/hewenhao/.codex/worktrees/single-channel-43-features/特征提取"
    )
    if not reference_root.exists():
        pytest.skip("accepted 43-feature reference worktree is unavailable")
    sys.path.insert(0, str(reference_root))
    try:
        from feature_pipeline.features import extract_features

        time = np.arange(4800, dtype=np.float64) / 12_000.0
        samples = (
            0.7 * np.sin(2.0 * np.pi * 24.6666666667 * time)
            + 0.2 * np.sin(2.0 * np.pi * 49.3333333333 * time)
            + 0.1 * np.sin(2.0 * np.pi * 320.0 * time)
        )
        accepted = extract_features(samples, rpm=1480.0)
        deployed = extract_catboost43_features(samples, rpm=1480.0)
    finally:
        sys.path.remove(str(reference_root))
    np.testing.assert_allclose(
        deployed.values,
        tuple(accepted[name] for name in FORMAL_FEATURE_NAMES),
        rtol=1e-12,
        atol=1e-12,
    )
