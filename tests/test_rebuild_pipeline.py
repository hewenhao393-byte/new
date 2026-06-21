from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.pipeline import METADATA_COLUMNS, extract_split_features, write_feature_dictionary


def test_extract_split_skips_nonfinite_run_and_reuses_file_cache(tmp_path: Path) -> None:
    raw_path = tmp_path / "Motor-4" / "70" / "正常状态1" / "正常状态1-通道4.csv"
    raw_path.parent.mkdir(parents=True)
    fs = 20_000
    t = np.arange(fs) / fs
    valid = np.sin(2 * np.pi * 1000 * t)
    invalid = valid.copy()
    invalid[100] = np.nan
    pd.DataFrame({"time": (np.arange(fs) + 1) / fs, "0": valid, "1": invalid}).to_csv(raw_path, index=False)

    config = PipelineConfig(data_root=tmp_path, output_root=tmp_path / "output")
    manifest = pd.DataFrame(
        [
            {
                "file_path": str(raw_path),
                "label": "正常",
                "raw_fault": "正常状态1",
                "rpm": 2070.0,
                "machine_id": "Motor-4",
                "condition_id": "Motor-4_70_正常状态1",
                "run_id": "Motor-4_70_正常状态1_run_0",
                "source_column": "0",
                "channel": 4,
                "original_fs": fs,
                "n_samples": fs,
                "duration_s": 1.0,
                "status": "indexed",
            },
            {
                "file_path": str(raw_path),
                "label": "正常",
                "raw_fault": "正常状态1",
                "rpm": 2070.0,
                "machine_id": "Motor-4",
                "condition_id": "Motor-4_70_正常状态1",
                "run_id": "Motor-4_70_正常状态1_run_1",
                "source_column": "1",
                "channel": 4,
                "original_fs": fs,
                "n_samples": fs,
                "duration_s": 1.0,
                "status": "indexed",
            },
        ]
    )

    result = extract_split_features(manifest, "train", config)
    first_mtime = result.feature_csv.stat().st_mtime_ns
    reused = extract_split_features(manifest, "train", config)

    features = pd.read_csv(result.feature_csv)
    quality = pd.read_csv(result.quality_csv)
    assert list(features.columns) == METADATA_COLUMNS + FEATURE_COLUMNS
    assert len(features) == 4
    assert features["run_id"].nunique() == 1
    assert np.isclose(features.iloc[0]["window_start"], 0.0)
    assert np.isclose(features.iloc[0]["window_end"], 4096 / 12000)
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    assert set(quality["run_id"]) == {"Motor-4_70_正常状态1_run_1"}
    assert set(quality["reason"]) == {"raw_nonfinite"}
    assert reused.feature_csv.stat().st_mtime_ns == first_mtime


def test_feature_dictionary_contains_all_candidate_features(tmp_path: Path) -> None:
    output = tmp_path / "feature_dictionary.csv"
    write_feature_dictionary(output)

    dictionary = pd.read_csv(output)
    assert len(dictionary) == 84
    assert dictionary["feature_name"].tolist() == FEATURE_COLUMNS
    assert dictionary["chinese_name"].str.len().gt(0).all()
    assert dictionary["feature_group"].nunique() == 5
