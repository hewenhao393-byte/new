from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_weight_study.contracts import FEATURE_NAMES
from fusion_weight_study.data import DataContractError, load_aligned_data, sha256_file


def _write_inputs(tmp_path: Path):
    paths = {}
    records = [("p1::0", "train_dev", "正常"), ("p2::0", "train_dev", "汽蚀"), ("p3::0", "test", "正常")]
    for channel in (3, 4, 5):
        rows = []
        for record_id, split, label in records:
            row = {
                "record_id": record_id, "group_id": record_id, "window_id": 0,
                "split_mode": "file", "split": split, "channel": channel,
                "measurement_position": f"CH{channel}", "motor": "m", "rpm": 1.0,
                "speed_percent": 100, "condition": "c", "state": "s",
                "severity": "unknown", "label": label, "start_sample": 0,
                "end_sample": 10, "start_time": 0.0, "end_time": 1.0,
                "train_block_id": "b", "quality_flag": "ok", "quality_reason": "",
            }
            row.update({name: float(i + channel) for i, name in enumerate(FEATURE_NAMES)})
            rows.append(row)
        path = tmp_path / f"ch{channel}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths[channel] = path
    split = pd.DataFrame([
        {"parent_group_id": "p1", "record_id": "p1::0", "label": "正常", "split": "train_dev"},
        {"parent_group_id": "p2", "record_id": "p2::0", "label": "汽蚀", "split": "train_dev"},
        {"parent_group_id": "p3", "record_id": "p3::0", "label": "正常", "split": "test"},
    ])
    split_path = tmp_path / "split.csv"; split.to_csv(split_path, index=False)
    folds = []
    for fold in (1, 2, 3):
        for parent in ("p1", "p2"):
            validation = (fold == 1 and parent == "p1") or (fold == 2 and parent == "p2")
            folds.append({"fold": fold, "parent_group_id": parent, "role": "validation" if validation else "fit"})
    # Ensure each parent is validation once; fold 3 intentionally has only fit rows.
    folds_path = tmp_path / "folds.csv"; pd.DataFrame(folds).to_csv(folds_path, index=False)
    return paths, split_path, folds_path


def test_loads_aligned_channels_and_records_hashes(tmp_path) -> None:
    paths, split, folds = _write_inputs(tmp_path)
    data = load_aligned_data(paths, split, folds, require_all_folds_nonempty=False)
    assert len(data.train_dev[3]) == 2
    assert len(data.strict[5]) == 1
    assert data.source_hashes[str(paths[3])] == sha256_file(paths[3])
    assert data.fold_by_parent == {"p1": 1, "p2": 2}


@pytest.mark.parametrize("defect", ["missing", "duplicate", "label", "nonfinite", "channel", "feature_order"])
def test_channel_contract_defects_are_hard_failures(tmp_path, defect) -> None:
    paths, split, folds = _write_inputs(tmp_path)
    frame = pd.read_csv(paths[4])
    if defect == "missing": frame = frame.iloc[:-1]
    elif defect == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif defect == "label": frame.loc[0, "label"] = "汽蚀"
    elif defect == "nonfinite": frame.loc[0, FEATURE_NAMES[0]] = np.nan
    elif defect == "channel": frame.loc[0, "channel"] = 3
    elif defect == "feature_order":
        columns = list(frame.columns); a, b = columns.index(FEATURE_NAMES[0]), columns.index(FEATURE_NAMES[1]); columns[a], columns[b] = columns[b], columns[a]; frame = frame[columns]
    frame.to_csv(paths[4], index=False)
    with pytest.raises(DataContractError):
        load_aligned_data(paths, split, folds, require_all_folds_nonempty=False)


def test_train_strict_parent_overlap_is_hard_failure(tmp_path) -> None:
    paths, split, folds = _write_inputs(tmp_path)
    split_frame = pd.read_csv(split)
    split_frame.loc[2, "parent_group_id"] = "p1"
    split_frame.to_csv(split, index=False)
    with pytest.raises(DataContractError, match="both train_dev and strict"):
        load_aligned_data(paths, split, folds, require_all_folds_nonempty=False)
