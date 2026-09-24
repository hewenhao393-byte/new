from __future__ import annotations

import pytest

from pump_fault_app.domain.diagnosis_models import WindowDiagnosisResult
from pump_fault_app.fusion import fuse_window_probabilities


def _window_prediction(
    window_index: int,
    predicted_label: str,
    probabilities: tuple[float, ...],
) -> WindowDiagnosisResult:
    return WindowDiagnosisResult(
        window_index=window_index,
        start_index=window_index * 2400,
        end_index=window_index * 2400 + 4800,
        predicted_label=predicted_label,
        class_probabilities=probabilities,
    )


def test_fuse_window_predictions_averages_probabilities_in_formal_label_order() -> None:
    first = _window_prediction(
        0,
        "正常",
        (0.60, 0.10, 0.10, 0.05, 0.10, 0.05),
    )
    second = _window_prediction(
        1,
        "转子不平衡",
        (0.20, 0.50, 0.10, 0.10, 0.05, 0.05),
    )

    result = fuse_window_probabilities("CH3", (first, second))

    assert result.class_probabilities == pytest.approx((0.40, 0.30, 0.10, 0.075, 0.075, 0.05))
    assert result.predicted_label == "正常"
    assert result.window_consistency == 0.5
    assert result.window_count == 2
    assert result.valid_window_count == 2


def test_fuse_window_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one window prediction is required"):
        fuse_window_probabilities("CH3", ())
