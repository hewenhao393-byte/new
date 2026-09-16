import pandas as pd
import pytest

from tools.record_level_single_channel_catboost import (
    build_record_assignment,
    split_distribution,
    validate_no_leakage,
)


RECORD_COLUMNS = [
    "group_id",
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]


@pytest.fixture
def example_records():
    rows = []
    specifications = [
        ("source/a_ch4.csv", 5, "正常", "Motor-2", 100),
        ("source/b_ch4.csv", 2, "汽蚀", "Motor-4", 70),
    ]
    for source_index, (ch4_path, count, label, device_id, speed) in enumerate(
        specifications
    ):
        for record_index in range(count):
            rows.append(
                {
                    "group_id": f"g{source_index}-{record_index}",
                    "record_column": str(record_index),
                    "label": label,
                    "device_id": device_id,
                    "speed_percent": speed,
                    "ch3_path": ch4_path.replace("ch4", "ch3"),
                    "ch4_path": ch4_path,
                    "ch5_path": ch4_path.replace("ch4", "ch5"),
                }
            )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


@pytest.fixture
def example_assignment(example_records):
    return build_record_assignment(example_records, test_fraction=0.2, seed=42)


@pytest.fixture
def example_windows(example_records):
    rows = []
    for record in example_records.to_dict("records"):
        for window_id in range(2):
            rows.append({**record, "window_id": window_id, "feature": window_id + 0.5})
    return pd.DataFrame(rows)


def test_split_keeps_every_record_and_is_deterministic(example_records):
    first = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    shuffled = example_records.sample(frac=1, random_state=7).reset_index(drop=True)
    second = build_record_assignment(shuffled, test_fraction=0.2, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["group_id"].is_unique
    assert set(first["group_id"]) == set(example_records["group_id"])


def test_each_eligible_file_contributes_train_and_test_records(example_records):
    assignment = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    counts = (
        assignment.groupby(["ch4_path", "subset"])["group_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    assert (counts["train"] > 0).all()
    assert (counts["test"] > 0).all()
    assert counts.loc["source/a_ch4.csv", "test"] == 1
    assert counts.loc["source/b_ch4.csv", "test"] == 1


def test_split_uses_requested_rounded_test_count(example_records):
    assignment = build_record_assignment(example_records, test_fraction=0.6, seed=42)
    counts = assignment.groupby(["ch4_path", "subset"])["group_id"].nunique()

    assert counts.loc[("source/a_ch4.csv", "test")] == 3
    assert counts.loc[("source/a_ch4.csv", "train")] == 2
    assert counts.loc[("source/b_ch4.csv", "test")] == 1
    assert counts.loc[("source/b_ch4.csv", "train")] == 1


def test_split_rejects_missing_required_columns(example_records):
    with pytest.raises(ValueError, match="Missing columns.*ch5_path"):
        build_record_assignment(example_records.drop(columns="ch5_path"))


def test_split_rejects_conflicting_group_metadata(example_records):
    conflicting = pd.concat(
        [example_records, example_records.iloc[[0]].assign(label="轴承故障")],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="group_id metadata"):
        build_record_assignment(conflicting)


def test_validate_returns_all_windows_with_subset(
    example_windows, example_assignment
):
    merged = validate_no_leakage(example_windows, example_assignment)

    assert len(merged) == len(example_windows)
    assert set(merged["subset"]) == {"train", "test"}
    assert merged[["group_id", "window_id"]].equals(
        example_windows[["group_id", "window_id"]]
    )


def test_validate_rejects_duplicate_group_assignment(example_windows, example_assignment):
    broken = pd.concat([example_assignment, example_assignment.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate group assignment"):
        validate_no_leakage(example_windows, broken)


def test_validate_rejects_missing_assignment(example_windows, example_assignment):
    broken = example_assignment.iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="no record assignment"):
        validate_no_leakage(example_windows, broken)


def test_validate_rejects_duplicate_window_key(example_windows, example_assignment):
    broken = pd.concat([example_windows, example_windows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate window key"):
        validate_no_leakage(broken, example_assignment)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("record_column", "wrong-record"),
        ("label", "轴承故障"),
        ("device_id", "wrong-device"),
        ("speed_percent", 999),
        ("ch3_path", "/wrong/channel3.csv"),
        ("ch4_path", "/wrong/channel4.csv"),
        ("ch5_path", "/wrong/channel5.csv"),
    ],
)
def test_validate_rejects_path_or_metadata_mismatch(
    example_windows, example_assignment, column, replacement
):
    broken = example_windows.copy()
    group_id = broken.loc[0, "group_id"]
    broken.loc[broken["group_id"].eq(group_id), column] = replacement

    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_no_leakage(broken, example_assignment)


def test_validate_rejects_unstable_channel_path_within_group(
    example_windows, example_assignment
):
    broken = example_windows.copy()
    broken.loc[0, "ch3_path"] = "/wrong/channel3.csv"

    with pytest.raises(ValueError, match="Inconsistent channel metadata"):
        validate_no_leakage(broken, example_assignment)


def test_validate_rejects_train_test_group_overlap(example_windows, example_assignment):
    overlapping = pd.concat(
        [
            example_assignment,
            example_assignment.iloc[[0]].assign(subset=lambda frame: frame["subset"].map(
                {"train": "test", "test": "train"}
            )),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="group leakage|Duplicate group assignment"):
        validate_no_leakage(example_windows, overlapping)


def test_split_distribution_counts_records_and_files(example_assignment):
    distribution = split_distribution(example_assignment)
    expected_columns = [
        "subset",
        "label",
        "device_id",
        "speed_percent",
        "records",
        "files",
    ]

    assert distribution.columns.tolist() == expected_columns
    assert distribution["records"].sum() == len(example_assignment)
    assert (distribution["files"] == 1).all()
