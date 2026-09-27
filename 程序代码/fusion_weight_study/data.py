from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import CHANNELS, CV_FOLDS, FEATURE_NAMES


class DataContractError(ValueError):
    """Raised when channel data cannot be aligned without leakage."""


WINDOW_KEY = ("record_id", "window_id", "start_sample", "end_sample")
MATCH_COLUMNS = WINDOW_KEY + ("label", "motor", "rpm", "speed_percent", "condition", "state", "severity")


@dataclass(frozen=True)
class AlignedData:
    train_dev: Mapping[int, pd.DataFrame]
    strict: Mapping[int, pd.DataFrame]
    fold_by_parent: Mapping[str, int]
    source_hashes: Mapping[str, str]
    split_frame: pd.DataFrame


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_from_record(record_id: str) -> str:
    if "::" not in str(record_id):
        raise DataContractError(f"record_id lacks parent suffix: {record_id!r}")
    return str(record_id).rsplit("::", 1)[0]


def _load_channel(path: Path, channel: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(WINDOW_KEY + MATCH_COLUMNS + ("channel",)) - set(frame.columns)
    if missing:
        raise DataContractError(f"CH{channel} missing columns: {sorted(missing)}")
    observed_features = tuple(column for column in frame.columns if column in FEATURE_NAMES)
    if observed_features != FEATURE_NAMES:
        raise DataContractError(f"CH{channel} feature order/count drift")
    if not frame["channel"].eq(channel).all():
        raise DataContractError(f"CH{channel} contains a different channel id")
    if frame.duplicated(list(WINDOW_KEY)).any():
        raise DataContractError(f"CH{channel} contains duplicate window keys")
    values = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError(f"CH{channel} contains non-finite features")
    frame = frame.copy()
    frame["parent_group_id"] = frame["record_id"].map(parent_from_record)
    return frame


def _validate_alignment(frames: Mapping[int, pd.DataFrame]) -> None:
    reference = frames[CHANNELS[0]].sort_values(list(WINDOW_KEY)).reset_index(drop=True)
    for channel in CHANNELS[1:]:
        candidate = frames[channel].sort_values(list(WINDOW_KEY)).reset_index(drop=True)
        if len(candidate) != len(reference):
            raise DataContractError(f"CH{channel} window count differs")
        for column in MATCH_COLUMNS:
            left = reference[column].astype(str).to_numpy()
            right = candidate[column].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise DataContractError(f"CH{channel} mismatch in {column}")


def _load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"parent_group_id", "record_id", "label", "split"}
    if not required.issubset(frame.columns):
        raise DataContractError(f"split file missing columns: {sorted(required - set(frame.columns))}")
    normalized = frame["split"].replace({"test": "strict"})
    if not set(normalized).issubset({"train_dev", "strict"}):
        raise DataContractError("unknown split value")
    frame = frame.copy(); frame["split"] = normalized
    by_parent = frame.groupby("parent_group_id")["split"].nunique()
    if (by_parent > 1).any():
        raise DataContractError("a parent appears in both train_dev and strict")
    return frame


def _load_fold_map(path: Path, train_parents: set[str], require_all_folds_nonempty: bool) -> dict[str, int]:
    folds = pd.read_csv(path, encoding="utf-8-sig")
    required = {"fold", "parent_group_id", "role"}
    if not required.issubset(folds.columns):
        raise DataContractError("fold file missing required columns")
    if not set(folds["role"]).issubset({"fit", "validation"}):
        raise DataContractError("unknown fold role")
    validation = folds.loc[folds["role"].eq("validation"), ["fold", "parent_group_id"]]
    counts = validation["parent_group_id"].value_counts()
    if set(counts.index) != train_parents or not counts.eq(1).all():
        raise DataContractError("each train_dev parent must be validation exactly once")
    if not set(folds["parent_group_id"]).issubset(train_parents):
        raise DataContractError("strict/unknown parent appears in CV folds")
    if require_all_folds_nonempty and set(validation["fold"].astype(int)) != set(range(1, CV_FOLDS + 1)):
        raise DataContractError("all configured validation folds must be non-empty")
    return dict(zip(validation["parent_group_id"], validation["fold"].astype(int)))


def load_aligned_data(
    channel_paths: Mapping[int, Path | str],
    split_path: Path | str,
    folds_path: Path | str,
    *,
    require_all_folds_nonempty: bool = True,
) -> AlignedData:
    if set(channel_paths) != set(CHANNELS):
        raise DataContractError("exactly CH3, CH4, and CH5 paths are required")
    paths = {channel: Path(channel_paths[channel]) for channel in CHANNELS}
    frames = {channel: _load_channel(paths[channel], channel) for channel in CHANNELS}
    _validate_alignment(frames)
    split = _load_split(Path(split_path))
    record_split = split[["record_id", "parent_group_id", "label", "split"]].drop_duplicates()
    if record_split["record_id"].duplicated().any():
        raise DataContractError("split file has duplicate/conflicting record rows")
    train_dev, strict = {}, {}
    for channel, frame in frames.items():
        merged = frame.merge(record_split, on="record_id", suffixes=("", "_contract"), validate="many_to_one")
        if len(merged) != len(frame):
            raise DataContractError("feature record is absent from split file")
        for column in ("parent_group_id", "label"):
            if not merged[column].astype(str).equals(merged[f"{column}_contract"].astype(str)):
                raise DataContractError(f"split mismatch in {column}")
        train_dev[channel] = merged.loc[merged["split_contract"].eq("train_dev")].copy()
        strict[channel] = merged.loc[merged["split_contract"].eq("strict")].copy()
    train_parents = set(train_dev[3]["parent_group_id"])
    strict_parents = set(strict[3]["parent_group_id"])
    if train_parents & strict_parents:
        raise DataContractError("a parent appears in both train_dev and strict")
    fold_map = _load_fold_map(Path(folds_path), train_parents, require_all_folds_nonempty)
    hashes = {str(path): sha256_file(path) for path in (*paths.values(), Path(split_path), Path(folds_path))}
    return AlignedData(train_dev, strict, fold_map, hashes, split)
