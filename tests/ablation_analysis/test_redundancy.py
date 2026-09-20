import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_40, FEATURE_43
from ablation_analysis.redundancy import consolidate_pairs, redundancy_decisions


def _pairs(rows):
    return pd.DataFrame(
        rows,
        columns=["feature_a", "feature_b", "pearson_r", "channel", "split_mode"],
    )


def test_reversed_pairs_are_consolidated_with_stable_summary_and_details():
    source = _pairs(
        [
            ("std", "rms", -0.96, 5, "temporal"),
            ("rms", "std", 0.99, 3, "record"),
        ]
    )

    result = consolidate_pairs(source, include_source_rows=True)

    assert result.to_dict("records") == [
        {
            "feature_a": "rms",
            "feature_b": "std",
            "occurrence_count": 2,
            "channels": [3, 5],
            "split_modes": ["record", "temporal"],
            "min_abs_r": pytest.approx(0.96),
            "max_abs_r": pytest.approx(0.99),
            "signs": ["negative", "positive"],
            "source_rows": [
                {"channel": 3, "split_mode": "record", "pearson_r": 0.99},
                {"channel": 5, "split_mode": "temporal", "pearson_r": -0.96},
            ],
        }
    ]


def test_output_order_is_deterministic_regardless_of_input_order():
    source = _pairs(
        [
            ("std", "rms", 0.98, 4, "record"),
            ("env_crest_factor", "env_kurtosis", 0.95, 3, "temporal"),
        ]
    )

    first = consolidate_pairs(source)
    second = consolidate_pairs(source.iloc[::-1].reset_index(drop=True))

    pd.testing.assert_frame_equal(first, second)
    assert first[["feature_a", "feature_b"]].values.tolist() == [
        ["rms", "std"],
        ["env_kurtosis", "env_crest_factor"],
    ]


def test_duplicate_source_identity_is_rejected_after_pair_canonicalization():
    source = _pairs(
        [
            ("rms", "std", 0.98, 3, "record"),
            ("std", "rms", 0.99, 3, "record"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate source identity"):
        consolidate_pairs(source)


@pytest.mark.parametrize(
    "row, message",
    [
        (("rms", "rms", 0.99, 3, "record"), "self-pair"),
        (("not_a_feature", "rms", 0.99, 3, "record"), "unknown feature"),
        (("rms", "std", np.inf, 3, "record"), "non-finite"),
        (("rms", "std", 0.949, 3, "record"), "below correlation threshold"),
        (("rms", "std", 0.99, 2, "record"), "invalid channel"),
        (("rms", "std", 0.99, 3, "random"), "invalid split_mode"),
    ],
)
def test_invalid_source_rows_are_rejected(row, message):
    with pytest.raises(ValueError, match=message):
        consolidate_pairs(_pairs([row]))


@pytest.mark.parametrize("include_source_rows", [False, True])
def test_empty_input_has_stable_columns(include_source_rows):
    result = consolidate_pairs(_pairs([]), include_source_rows=include_source_rows)

    expected = [
        "feature_a",
        "feature_b",
        "occurrence_count",
        "channels",
        "split_modes",
        "min_abs_r",
        "max_abs_r",
        "signs",
    ]
    if include_source_rows:
        expected.append("source_rows")
    assert result.empty
    assert result.columns.tolist() == expected


def test_redundancy_decisions_preserve_feature_order_and_exact_removals():
    decisions = redundancy_decisions(FEATURE_43)

    assert decisions["feature"].tolist() == FEATURE_43
    assert decisions.shape[0] == 43
    removed = decisions.loc[decisions["decision"].eq("remove"), "feature"].tolist()
    retained = decisions.loc[decisions["decision"].eq("retain"), "feature"].tolist()
    assert removed == ["std", "band_energy_3000_5000_ratio", "wp_energy_ratio_7"]
    assert retained == FEATURE_40


def test_all_decisions_have_category_reason_and_review_note():
    decisions = redundancy_decisions(FEATURE_43)

    assert decisions[["category", "decision", "reason", "review_note"]].notna().all().all()
    assert decisions[["category", "decision", "reason", "review_note"]].apply(
        lambda column: column.str.strip().ne("").all()
    ).all()
    reasons = decisions.set_index("feature")["reason"]
    assert reasons["std"] == "near-duplicate of RMS for these vibration windows"
    assert reasons["band_energy_3000_5000_ratio"] == "reference component of four-part closed composition"
    assert reasons["wp_energy_ratio_7"] == "reference component of eight-part closed composition"


def test_locally_correlated_families_are_explicitly_retained_for_instability():
    decisions = redundancy_decisions(FEATURE_43).set_index("feature")
    locally_correlated = [
        "crest_factor",
        "impulse_factor",
        "clearance_factor",
        "shape_factor",
        "spectral_centroid",
        "high_low_energy_ratio",
        "env_kurtosis",
        "env_crest_factor",
        "env_spectral_entropy",
        "env_peak_energy_ratio",
        "env_peak_concentration",
        "env_peak_count",
    ]

    assert decisions.loc[locally_correlated, "decision"].eq("retain").all()
    assert decisions.loc[locally_correlated, "review_note"].str.contains(
        "correlation not stable across all channels", regex=False
    ).all()


def test_redundancy_decisions_reject_non_contract_feature_sequence():
    with pytest.raises(ValueError, match="exactly FEATURE_43"):
        redundancy_decisions(FEATURE_43[:-1])
