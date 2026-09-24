from __future__ import annotations

import pytest
pytest.skip("superseded by V3 entrypoint tests during staged migration", allow_module_level=True)

from pathlib import Path

import pandas as pd
import pytest

from pump_fault_app.batch import (
    load_batch_manifest,
    run_batch_inference_from_manifest,
)
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv

import numpy as np


def test_load_batch_manifest_reads_required_and_optional_columns(tmp_path: Path) -> None:
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0))

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "file_path": str(signal_path),
                "sampling_rate_hz": 12000,
                "rpm": 1500.0,
                "signal_column": "通道4",
                "time_column": "time",
                "device_id": "Motor-2",
                "measurement_position": "泵端",
            }
        ]
    ).to_csv(manifest_path, index=False)

    manifest = load_batch_manifest(manifest_path)

    assert len(manifest.items) == 1
    item = manifest.items[0]
    assert item.file_path == signal_path
    assert item.sampling_rate_hz == 12000
    assert item.rpm == 1500.0
    assert item.signal_column == "通道4"
    assert item.time_column == "time"
    assert item.device_id == "Motor-2"
    assert item.measurement_position == "泵端"


def test_load_batch_manifest_rejects_missing_required_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"file_path": "a.csv", "rpm": 1500.0}]).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest is missing required columns"):
        load_batch_manifest(manifest_path)


def test_load_batch_manifest_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(columns=["file_path", "sampling_rate_hz", "rpm"]).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest must contain at least one row"):
        load_batch_manifest(manifest_path)


def test_load_batch_manifest_rejects_blank_file_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"file_path": "   ", "sampling_rate_hz": 12000, "rpm": 1500.0}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest row 1 has empty file_path"):
        load_batch_manifest(manifest_path)


def test_load_batch_manifest_rejects_nonpositive_sampling_rate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"file_path": "a.csv", "sampling_rate_hz": 0, "rpm": 1500.0}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest row 1 has invalid sampling_rate_hz"):
        load_batch_manifest(manifest_path)


def test_load_batch_manifest_rejects_nonpositive_rpm(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"file_path": "a.csv", "sampling_rate_hz": 12000, "rpm": -1.0}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest row 1 has invalid rpm"):
        load_batch_manifest(manifest_path)


def test_run_batch_inference_from_manifest_uses_per_file_parameters(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    _write_signal_csv(
        good_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    bad_path = tmp_path / "bad.csv"
    _write_signal_csv(bad_path, np.ones(1200, dtype=np.float64))

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "file_path": str(good_path),
                "sampling_rate_hz": 12000,
                "rpm": 1500.0,
                "signal_column": "通道4",
                "time_column": "time",
            },
            {
                "file_path": str(bad_path),
                "sampling_rate_hz": 12000,
                "rpm": 1500.0,
            },
        ]
    ).to_csv(manifest_path, index=False)

    manifest = load_batch_manifest(manifest_path)
    result = run_batch_inference_from_manifest(manifest, model_bundle_path=bundle_path)

    assert result.total_count == 2
    assert result.diagnosed_count == 1
    assert result.rejected_count == 1
    assert result.summaries[0].status == "diagnosed"
    assert result.summaries[1].status == "rejected"
