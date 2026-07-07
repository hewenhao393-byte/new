from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_fault_app.services import (
    AppBatchRunRequest,
    AppSingleRunRequest,
    run_batch_diagnosis,
    run_single_diagnosis,
)
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_run_single_diagnosis_returns_summary_and_optional_export(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(
        signal_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            export_root=tmp_path / "single_exports",
        )
    )

    assert result.summary.success is True
    assert result.summary.status == "diagnosed"
    assert result.export_result is not None
    assert result.export_result.json_path.exists()
    assert result.export_result.csv_path.exists()


def test_run_batch_diagnosis_supports_manifest_and_export(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    bad_path = tmp_path / "bad.csv"
    _write_signal_csv(good_path, 0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0))
    _write_signal_csv(bad_path, np.ones(1200, dtype=np.float64))

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"file_path": str(good_path), "sampling_rate_hz": 12000, "rpm": 1500.0},
            {"file_path": str(bad_path), "sampling_rate_hz": 12000, "rpm": 1500.0},
        ]
    ).to_csv(manifest_path, index=False)

    result = run_batch_diagnosis(
        AppBatchRunRequest(
            manifest_path=manifest_path,
            model_bundle_path=bundle_path,
            export_root=tmp_path / "batch_exports",
        )
    )

    assert result.batch_result.total_count == 2
    assert result.batch_result.diagnosed_count == 1
    assert result.batch_result.rejected_count == 1
    assert result.export_result is not None
    assert result.export_result.json_path.exists()
    assert result.export_result.csv_path.exists()


def test_run_batch_diagnosis_rejects_mixed_manifest_and_file_list(tmp_path: Path) -> None:
    try:
        run_batch_diagnosis(
            AppBatchRunRequest(
                file_paths=(tmp_path / "a.csv",),
                manifest_path=tmp_path / "manifest.csv",
                sampling_rate_hz=12000,
                rpm=1500.0,
            )
        )
    except ValueError as exc:
        assert "exactly one of file_paths or manifest_path must be provided" in str(exc)
    else:
        raise AssertionError("expected mixed batch request to raise ValueError")
