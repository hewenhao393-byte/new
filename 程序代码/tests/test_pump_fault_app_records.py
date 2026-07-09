from __future__ import annotations

import pytest

from pump_fault_app.domain.records import FeatureVector, SampleMetadata, TimeDomainSeries


def test_feature_vector_exposes_feature_map() -> None:
    vector = FeatureVector(
        feature_names=("kurtosis", "skewness"),
        values=(3.2, 0.4),
        metadata=SampleMetadata(label="正常", rpm=1480.0, channel=4),
    )

    payload = vector.as_dict()

    assert payload["metadata"]["label"] == "正常"
    assert payload["feature_map"] == {"kurtosis": 3.2, "skewness": 0.4}


def test_feature_vector_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="feature name and value counts must match"):
        FeatureVector(
            feature_names=("kurtosis", "skewness"),
            values=(3.2,),
            metadata=SampleMetadata(label="正常"),
        )


def test_time_domain_series_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="time axis and amplitude lengths must match"):
        TimeDomainSeries(
            time_s=(0.0, 0.1),
            amplitude=(1.0,),
            sampling_rate_hz=12000,
            point_count=2,
            downsampled_for_display=False,
        )
