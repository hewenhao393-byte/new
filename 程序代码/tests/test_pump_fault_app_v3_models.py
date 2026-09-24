from pathlib import Path

import pytest

from pump_fault_app.domain.diagnosis_models import (
    ChannelDiagnosisResult,
    ChannelInput,
    MultiChannelDiagnosisResult,
    MultiChannelInferenceRequest,
    probability_tuple,
)


def test_request_rejects_duplicate_channels() -> None:
    item = ChannelInput("CH3", Path("a.csv"), "signal", None)
    with pytest.raises(ValueError, match="duplicate channel"):
        MultiChannelInferenceRequest((item, item), 12_000, 1500.0)


def test_probability_tuple_requires_six_normalized_values() -> None:
    assert probability_tuple([0.1, 0.2, 0.3, 0.1, 0.2, 0.1]) == pytest.approx(
        (0.1, 0.2, 0.3, 0.1, 0.2, 0.1)
    )
    with pytest.raises(ValueError, match="sum to 1"):
        probability_tuple([0.1] * 6)


def test_invalid_channel_cannot_contain_a_prediction() -> None:
    with pytest.raises(ValueError, match="invalid channel"):
        ChannelDiagnosisResult(
            channel="CH3",
            status="invalid",
            failure_stage="quality",
            failure_message="all zero",
            predicted_label="正常",
            class_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


def test_failed_multichannel_result_cannot_contain_a_final_label() -> None:
    with pytest.raises(ValueError, match="failed result"):
        MultiChannelDiagnosisResult(
            status="failed",
            input_channels=("CH3",),
            valid_channels=(),
            invalid_channels=("CH3",),
            channel_results=(),
            predicted_label="正常",
            fused_probabilities=None,
            agreement_level=None,
            agreement_count=0,
            valid_channel_count=0,
        )
