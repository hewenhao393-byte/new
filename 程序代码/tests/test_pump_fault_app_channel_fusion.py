import pytest

from pump_fault_app.domain.diagnosis_models import ChannelDiagnosisResult
from pump_fault_app.fusion import fuse_valid_channels


def _valid(channel: str, label: str, probabilities: tuple[float, ...]) -> ChannelDiagnosisResult:
    return ChannelDiagnosisResult(
        channel=channel,
        status="valid",
        predicted_label=label,
        class_probabilities=probabilities,
    )


def _invalid(channel: str) -> ChannelDiagnosisResult:
    return ChannelDiagnosisResult(
        channel=channel,
        status="invalid",
        failure_stage="quality",
        failure_message="all zero",
    )


def test_two_channel_fusion_is_equal_probability_mean() -> None:
    ch3 = _valid("CH3", "正常", (0.6, 0.1, 0.1, 0.1, 0.05, 0.05))
    ch4 = _valid("CH4", "正常", (0.4, 0.2, 0.1, 0.1, 0.1, 0.1))
    result = fuse_valid_channels((ch3, ch4))
    assert result.fused_probabilities == pytest.approx((0.5, 0.15, 0.1, 0.1, 0.075, 0.075))
    assert result.agreement_level == "高一致"
    assert result.agreement_count == 2


def test_invalid_channel_is_excluded() -> None:
    ch4 = _valid("CH4", "正常", (0.6, 0.1, 0.1, 0.1, 0.05, 0.05))
    ch5 = _valid("CH5", "轴承故障", (0.1, 0.1, 0.1, 0.1, 0.5, 0.1))
    result = fuse_valid_channels((_invalid("CH3"), ch4, ch5))
    assert result.valid_channels == ("CH4", "CH5")
    assert result.invalid_channels == ("CH3",)
    assert result.agreement_level == "低一致"


def test_three_channel_majority_is_medium_agreement() -> None:
    result = fuse_valid_channels(
        (
            _valid("CH3", "汽蚀", (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)),
            _valid("CH4", "汽蚀", (0.1, 0.1, 0.1, 0.1, 0.2, 0.4)),
            _valid("CH5", "正常", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)),
        )
    )
    assert result.agreement_level == "中等一致"
    assert result.agreement_count == 2


def test_all_invalid_channels_return_failed() -> None:
    result = fuse_valid_channels((_invalid("CH3"), _invalid("CH4"), _invalid("CH5")))
    assert result.status == "failed"
    assert result.predicted_label is None
    assert result.fused_probabilities is None
