import numpy as np
import pandas as pd
import pytest

from ablation_analysis.confusion_diagnostics import (
    assign_target_group,
    cliffs_delta,
    effect_size_table,
    grouped_feature_summary,
    record_feature_medians,
    targeted_error_concentration,
)


@pytest.mark.parametrize(
    "label,predicted,expected",
    [
        ("松动", "松动", "correct_looseness"),
        ("松动", "轴承故障", "looseness_to_bearing"),
        ("轴承故障", "轴承故障", "correct_bearing"),
        ("轴承故障", "松动", "bearing_to_looseness"),
        ("正常", "正常", None),
        ("松动", "不平衡", None),
    ],
)
def test_assign_target_group_exact_mapping(label, predicted, expected):
    result = assign_target_group(pd.Series([label], index=["row"]), pd.Series([predicted], index=["row"]))
    assert result.index.tolist() == ["row"]
    assert result.iloc[0] == expected or (expected is None and pd.isna(result.iloc[0]))


def test_assign_target_group_vectorized_mapping_preserves_index():
    index = pd.Index([10, 20, 30, 40, 50], name="source_row")
    actual = pd.Series(["松动", "松动", "轴承故障", "轴承故障", "正常"], index=index)
    predicted = pd.Series(["松动", "轴承故障", "轴承故障", "松动", "正常"], index=index)

    result = assign_target_group(actual, predicted)

    expected = pd.Series(
        ["correct_looseness", "looseness_to_bearing", "correct_bearing", "bearing_to_looseness", None],
        index=index,
        name="target_group",
        dtype=object,
    )
    pd.testing.assert_series_equal(result, expected)


def test_grouped_summary_reports_window_record_and_quartile_statistics():
    frame = pd.DataFrame(
        {
            "target_group": ["g", "g", "g", "g"],
            "record_id": ["r1", "r1", "r2", "r2"],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = grouped_feature_summary(frame, ["f1"]).iloc[0]

    assert result["window_count"] == 4
    assert result["record_count"] == 2
    assert result["median"] == 2.5
    assert result["q1"] == 1.75
    assert result["q3"] == 3.25
    assert result["iqr"] == 1.5


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_grouped_summary_rejects_nonfinite_values(bad):
    frame = pd.DataFrame({"target_group": ["g"], "record_id": ["r"], "f1": [bad]})
    with pytest.raises(ValueError, match="non-finite"):
        grouped_feature_summary(frame, ["f1"])


def test_record_feature_medians_balance_many_windows_against_one_record():
    frame = pd.DataFrame(
        {
            "target_group": ["g"] * 101,
            "record_id": ["many"] * 100 + ["one"],
            "f1": [1.0] * 100 + [9.0],
        }
    )

    result = record_feature_medians(frame, ["f1"])

    assert len(result) == 2
    assert result.set_index("record_id")["f1"].to_dict() == {"many": 1.0, "one": 9.0}


def test_cliffs_delta_direction_magnitude_and_ties():
    assert cliffs_delta([3, 4], [1, 2]) == {"delta": 1.0, "abs_delta": 1.0, "magnitude": "large"}
    assert cliffs_delta([1, 2], [3, 4]) == {"delta": -1.0, "abs_delta": 1.0, "magnitude": "large"}
    assert cliffs_delta([1, 2], [1, 2]) == {
        "delta": 0.0,
        "abs_delta": 0.0,
        "magnitude": "negligible",
    }
    assert cliffs_delta([1, 2], [1, 3])["delta"] == -0.25


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1] * 85 + [2] * 15, [1] * 100, "small"),
        ([1] * 66 + [2] * 34, [1] * 100, "medium"),
        ([1] * 52 + [2] * 48, [1] * 100, "large"),
    ],
)
def test_cliffs_delta_magnitude_threshold_bands(x, y, expected):
    assert cliffs_delta(x, y)["magnitude"] == expected


@pytest.mark.parametrize(
    "absolute,expected",
    [(0.146999, "negligible"), (0.147, "small"), (0.33, "medium"), (0.474, "large")],
)
def test_cliffs_delta_magnitude_exact_boundaries(absolute, expected):
    from ablation_analysis.confusion_diagnostics import _cliffs_magnitude

    assert _cliffs_magnitude(absolute) == expected


@pytest.mark.parametrize("x,y", [([], [1]), ([1], []), ([np.nan], [1]), ([1], [np.inf])])
def test_cliffs_delta_rejects_empty_or_nonfinite_inputs(x, y):
    with pytest.raises(ValueError, match="empty|non-finite"):
        cliffs_delta(x, y)


def test_effect_size_table_emits_exactly_three_comparisons_and_flags_small_samples():
    rows = []
    groups = {
        "correct_looseness": [5, 6, 7, 8, 9],
        "looseness_to_bearing": [1, 2, 3, 4],
        "correct_bearing": [10, 11, 12, 13, 14],
        "bearing_to_looseness": [15, 16, 17, 18, 19],
    }
    for group, values in groups.items():
        rows.extend(
            {"target_group": group, "record_id": f"{group}-{i}", "f1": value}
            for i, value in enumerate(values)
        )

    result = effect_size_table(pd.DataFrame(rows), ["f1"])

    assert list(zip(result["group_a"], result["group_b"])) == [
        ("correct_looseness", "looseness_to_bearing"),
        ("correct_bearing", "bearing_to_looseness"),
        ("correct_looseness", "correct_bearing"),
    ]
    assert len(result) == 3
    assert result.iloc[0]["n_records_a"] == 5
    assert result.iloc[0]["n_records_b"] == 4
    assert bool(result.iloc[0]["small_sample"])
    assert result.iloc[0]["delta"] == 1.0
    assert not bool(result.iloc[1]["small_sample"])


def test_effect_size_table_balances_windows_to_records_before_comparison():
    frame = pd.DataFrame(
        [
            *(
                {"target_group": "correct_looseness", "record_id": "many", "f1": 1.0}
                for _ in range(100)
            ),
            {"target_group": "correct_looseness", "record_id": "one", "f1": 9.0},
            {"target_group": "looseness_to_bearing", "record_id": "l2b", "f1": 5.0},
            {"target_group": "correct_bearing", "record_id": "cb", "f1": 6.0},
            {"target_group": "bearing_to_looseness", "record_id": "b2l", "f1": 7.0},
        ]
    )

    result = effect_size_table(frame, ["f1"])

    first = result.iloc[0]
    assert first["n_records_a"] == 2
    assert first["n_records_b"] == 1
    assert first["delta"] == 0.0


def test_targeted_error_concentration_counts_records_and_top5_share_with_metadata():
    rows = []
    for index, count in enumerate([6, 5, 4, 3, 2, 1]):
        rows.extend(
            {
                "target_group": "looseness_to_bearing" if index % 2 == 0 else "bearing_to_looseness",
                "record_id": f"r{index}",
                "motor": "M2",
                "rpm": 740,
                "condition": "c",
                "state": "s",
                "severity": index,
            }
            for _ in range(count)
        )
    rows.append({"target_group": "correct_looseness", "record_id": "ignored", "motor": "M2"})

    per_record, summary = targeted_error_concentration(pd.DataFrame(rows))

    assert per_record["record_id"].tolist() == ["r0", "r1", "r2", "r3", "r4", "r5"]
    assert per_record["error_windows"].tolist() == [6, 5, 4, 3, 2, 1]
    assert {"target_group", "motor", "rpm", "condition", "state", "severity"}.issubset(per_record.columns)
    assert summary == {
        "total_error_windows": 21,
        "error_records": 6,
        "top5_error_windows": 20,
        "top5_share": pytest.approx(20 / 21),
    }


@pytest.mark.parametrize("missing", ["record_id", "motor", "rpm", "condition", "state", "severity"])
def test_targeted_error_concentration_requires_all_record_metadata(missing):
    frame = pd.DataFrame(
        {
            "target_group": ["looseness_to_bearing"],
            "record_id": ["r1"],
            "motor": ["M2"],
            "rpm": [740],
            "condition": ["c"],
            "state": ["s"],
            "severity": [1],
        }
    ).drop(columns=missing)

    with pytest.raises(ValueError, match=rf"missing required columns.*{missing}"):
        targeted_error_concentration(frame)


def test_targeted_error_concentration_rejects_conflicting_record_metadata_including_missing():
    frame = pd.DataFrame(
        {
            "target_group": ["looseness_to_bearing", "looseness_to_bearing"],
            "record_id": ["r1", "r1"],
            "motor": ["M2", "M2"],
            "rpm": [740, 740],
            "condition": ["c", "c"],
            "state": ["s", "s"],
            "severity": [1, np.nan],
        }
    )

    with pytest.raises(ValueError, match=r"conflicting metadata.*r1"):
        targeted_error_concentration(frame)


def test_targeted_error_concentration_rejects_null_record_id():
    frame = pd.DataFrame(
        {
            "target_group": ["bearing_to_looseness"],
            "record_id": [None],
            "motor": ["M2"],
            "rpm": [740],
            "condition": ["c"],
            "state": ["s"],
            "severity": [1],
        }
    )

    with pytest.raises(ValueError, match="null record_id"):
        targeted_error_concentration(frame)
