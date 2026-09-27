from __future__ import annotations

import itertools

import numpy as np
import pytest

from fusion_weight_study.contracts import CHANNELS, LABELS
from fusion_weight_study.fusion import (
    FusionError,
    derive_classwise_weights,
    derive_global_weights,
    fuse_probabilities,
)


PROBS = {
    3: np.array([[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]]),
    4: np.array([[0.1, 0.6, 0.1, 0.1, 0.05, 0.05]]),
    5: np.array([[0.1, 0.1, 0.6, 0.1, 0.05, 0.05]]),
}


@pytest.mark.parametrize("size", [1, 2, 3])
def test_all_channel_subsets_produce_normalized_probabilities(size) -> None:
    for subset in itertools.combinations(CHANNELS, size):
        probabilities = {channel: PROBS[channel] for channel in subset}
        for method in ("equal", "global", "classwise"):
            weights = None
            if method == "global": weights = {3: 1.0, 4: 2.0, 5: 3.0}
            if method == "classwise": weights = {channel: {label: channel for label in LABELS} for channel in CHANNELS}
            fused = fuse_probabilities(probabilities, method, weights)
            assert fused.shape == (1, 6)
            assert np.isfinite(fused).all() and (fused >= 0).all()
            np.testing.assert_allclose(fused.sum(axis=1), 1.0)


def test_global_weights_are_macro_f1_normalized() -> None:
    assert derive_global_weights({3: 0.2, 4: 0.3, 5: 0.5}) == {3: 0.2, 4: 0.3, 5: 0.5}
    assert derive_global_weights({3: 0.0, 4: 0.0, 5: 0.0}) == {3: 1 / 3, 4: 1 / 3, 5: 1 / 3}


def test_classwise_weights_use_f1_and_fallback_when_class_scores_are_zero() -> None:
    scores = {channel: {label: 0.0 for label in LABELS} for channel in CHANNELS}
    scores[3][LABELS[0]], scores[4][LABELS[0]], scores[5][LABELS[0]] = 1.0, 2.0, 1.0
    weights = derive_classwise_weights(scores)
    assert [weights[c][LABELS[0]] for c in CHANNELS] == [0.25, 0.5, 0.25]
    assert [weights[c][LABELS[1]] for c in CHANNELS] == pytest.approx([1 / 3] * 3)


def test_classwise_scores_are_renormalized_across_classes() -> None:
    weights = {channel: {label: float(channel + i) for i, label in enumerate(LABELS)} for channel in CHANNELS}
    fused = fuse_probabilities(PROBS, "classwise", weights)
    np.testing.assert_allclose(fused.sum(axis=1), [1.0])


@pytest.mark.parametrize("bad", [np.array([0, 1, 2]), np.ones((2, 5)), np.array([[1, 0, 0, 0, 0, -1]])])
def test_hard_labels_or_invalid_probabilities_are_rejected(bad) -> None:
    with pytest.raises(FusionError):
        fuse_probabilities({3: bad}, "equal")
