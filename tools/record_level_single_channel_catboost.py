"""Leakage-safe record assignment utilities for single-channel experiments.

This module intentionally contains no model training. It defines the shared
record-level split used by channels 3, 4, and 5 and validates that all sibling
channels and overlapping windows remain in the same subset.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


RECORD_METADATA_COLUMNS = [
    "group_id",
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]
MATCH_COLUMNS = [
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]
CHANNEL_PATH_COLUMNS = ["ch3_path", "ch4_path", "ch5_path"]


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {frame_name}: {missing}")


def _stable_local_seed(seed: int, source_path: object) -> int:
    digest = hashlib.sha256(f"{seed}:{source_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_record_assignment(
    records: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign complete 12-second records to train/test within each source file."""
    _require_columns(records, RECORD_METADATA_COLUMNS, "records")
    if not 0 <= test_fraction <= 1:
        raise ValueError("test_fraction must be between 0 and 1")

    metadata = records.loc[:, RECORD_METADATA_COLUMNS].drop_duplicates().copy()
    conflicting = metadata["group_id"].duplicated(keep=False)
    if conflicting.any():
        group_ids = sorted(metadata.loc[conflicting, "group_id"].astype(str).unique())
        raise ValueError(f"group_id metadata is not unique: {group_ids}")

    unique = metadata.sort_values(["ch4_path", "group_id"], kind="stable")
    parts = []
    for source_path, group in unique.groupby("ch4_path", sort=True, dropna=False):
        group = group.sort_values("group_id", kind="stable").reset_index(drop=True)
        rng = np.random.default_rng(_stable_local_seed(seed, source_path))
        shuffled = group.iloc[rng.permutation(len(group))].copy()
        test_count = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        shuffled["subset"] = "train"
        if test_count:
            shuffled.iloc[:test_count, shuffled.columns.get_loc("subset")] = "test"
        parts.append(shuffled)

    columns = RECORD_METADATA_COLUMNS + ["subset"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
        .loc[:, columns]
    )


def _equal_including_missing(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def validate_no_leakage(
    windows: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Validate record/window isolation and attach each window's subset."""
    _require_columns(windows, RECORD_METADATA_COLUMNS + ["window_id"], "windows")
    _require_columns(assignment, RECORD_METADATA_COLUMNS + ["subset"], "assignment")

    if assignment["group_id"].duplicated().any():
        raise ValueError("Duplicate group assignment causes group leakage")
    invalid_subsets = set(assignment["subset"].dropna()).difference({"train", "test"})
    if invalid_subsets or assignment["subset"].isna().any():
        raise ValueError(f"Invalid subset values: {sorted(map(str, invalid_subsets))}")
    if windows.duplicated(["group_id", "window_id"]).any():
        raise ValueError("Duplicate window key")

    path_counts = windows.groupby("group_id", dropna=False)[CHANNEL_PATH_COLUMNS].nunique(
        dropna=False
    )
    if not path_counts.eq(1).all().all():
        raise ValueError("Inconsistent channel metadata within group")

    train_ids = set(assignment.loc[assignment["subset"].eq("train"), "group_id"])
    test_ids = set(assignment.loc[assignment["subset"].eq("test"), "group_id"])
    if train_ids.intersection(test_ids):
        raise ValueError("group leakage detected between train and test")

    assignment_columns = RECORD_METADATA_COLUMNS + ["subset"]
    merged = windows.merge(
        assignment.loc[:, assignment_columns],
        on="group_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "__assignment"),
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        missing_ids = sorted(
            merged.loc[merged["_merge"].ne("both"), "group_id"].astype(str).unique()
        )
        raise ValueError(f"Some windows have no record assignment: {missing_ids}")

    for column in MATCH_COLUMNS:
        assigned_column = f"{column}__assignment"
        matches = _equal_including_missing(merged[column], merged[assigned_column])
        if not matches.all():
            group_ids = sorted(merged.loc[~matches, "group_id"].astype(str).unique())
            raise ValueError(f"Record metadata mismatch for {column}: {group_ids}")

    drop_columns = ["_merge"] + [f"{column}__assignment" for column in MATCH_COLUMNS]
    return merged.drop(columns=drop_columns)


def split_distribution(assignment: pd.DataFrame) -> pd.DataFrame:
    """Summarize record and source-file coverage for every split stratum."""
    required = [
        "subset",
        "label",
        "device_id",
        "speed_percent",
        "group_id",
        "ch4_path",
    ]
    _require_columns(assignment, required, "assignment")
    return (
        assignment.groupby(
            ["subset", "label", "device_id", "speed_percent"],
            dropna=False,
            sort=True,
        )
        .agg(records=("group_id", "nunique"), files=("ch4_path", "nunique"))
        .reset_index()
    )
