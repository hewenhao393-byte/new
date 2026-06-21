from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from config import PipelineConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.pipeline import METADATA_COLUMNS
from pump_diagnosis.plots import generate_stage1_plots
from pump_diagnosis.runner import validate_feature_csv


LABELS = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]


def test_config_snapshot_is_json_round_trip_stable() -> None:
    snapshot = PipelineConfig().as_serializable_dict()
    assert json.loads(json.dumps(snapshot, ensure_ascii=False)) == snapshot


def test_importing_runner_does_not_import_matplotlib() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pump_diagnosis.runner; print('matplotlib' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def _valid_feature_row(label: str = "正常") -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample_0",
        "label": label,
        "run_id": "run_0",
        "file_path": "/tmp/raw.csv",
        "machine_id": "Motor-4",
        "condition_id": "condition_0",
        "rpm": 2070.0,
        "channel": 4,
        "window_index": 0,
        "window_start": 0.0,
        "window_end": 4096 / 12000,
        "original_fs": 20_000,
        "processed_fs": 12_000,
    }
    row.update({name: 0.1 for name in FEATURE_COLUMNS})
    for name in ("aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"):
        row[f"wp_energy_{name}"] = 0.125
    return row


def test_validate_feature_csv_checks_schema_finiteness_and_wavelet_sum(tmp_path: Path) -> None:
    config = PipelineConfig(output_root=tmp_path)
    path = tmp_path / "features.csv"
    pd.DataFrame([_valid_feature_row()]).to_csv(path, index=False)

    summary = validate_feature_csv(path, config)

    assert summary["rows"] == 1
    assert summary["feature_count"] == 84
    assert summary["labels"] == {"正常": 1}

    broken = _valid_feature_row()
    broken["rms"] = np.nan
    pd.DataFrame([broken]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="NaN或Inf"):
        validate_feature_csv(path, config)


def test_generate_stage1_plots_creates_four_nonempty_images(tmp_path: Path) -> None:
    config = PipelineConfig(data_root=tmp_path, output_root=tmp_path / "output", plot_dpi=80)
    manifest_rows = []
    feature_rows = []
    for index, label in enumerate(LABELS):
        raw_path = tmp_path / f"class_{index}_通道4.csv"
        t = np.arange(20_000) / 20_000
        pd.DataFrame(
            {
                "time": (np.arange(20_000) + 1) / 20_000,
                "0": np.sin(2 * np.pi * (500 + index * 100) * t),
            }
        ).to_csv(raw_path, index=False)
        manifest_rows.append(
            {
                "file_path": str(raw_path),
                "label": label,
                "run_id": f"run_{index}",
                "source_column": "0",
            }
        )
        row = _valid_feature_row(label)
        row["sample_id"] = f"sample_{index}"
        row["run_id"] = f"run_{index}"
        row["file_path"] = str(raw_path)
        feature_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    feature_csv = tmp_path / "train_features.csv"
    pd.DataFrame(feature_rows, columns=METADATA_COLUMNS + FEATURE_COLUMNS).to_csv(feature_csv, index=False)

    paths = generate_stage1_plots(manifest, manifest, feature_csv, config)

    assert set(paths) == {"class_counts", "filter_response", "preprocessing_spectra", "wavelet_energy"}
    assert all(path.exists() and path.stat().st_size > 1000 for path in paths.values())
