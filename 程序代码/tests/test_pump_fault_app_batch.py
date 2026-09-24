from __future__ import annotations

import pytest
pytest.skip("superseded by V3 entrypoint tests during staged migration", allow_module_level=True)

from pathlib import Path

import numpy as np

from pump_fault_app.batch import BatchInferenceRequest, run_batch_inference
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_run_batch_inference_returns_summary_counts_and_items(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    good_samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
        + 0.05 * np.random.default_rng(17).normal(size=4800)
    )
    _write_signal_csv(good_path, good_samples)

    short_path = tmp_path / "short.csv"
    _write_signal_csv(short_path, np.ones(1200, dtype=np.float64))

    missing_path = tmp_path / "missing.csv"

    result = run_batch_inference(
        BatchInferenceRequest(
            file_paths=(good_path, short_path, missing_path),
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    assert result.total_count == 3
    assert result.success_count == 1
    assert result.failure_count == 2
    assert result.diagnosed_count == 1
    assert result.rejected_count == 1
    assert result.input_error_count == 1
    assert [summary.file_name for summary in result.summaries] == ["good.csv", "short.csv", None]
    assert result.summaries[0].status == "diagnosed"
    assert result.summaries[1].status == "rejected"
    assert result.summaries[2].status == "input_error"


def test_run_batch_inference_rejects_empty_file_list(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    try:
        run_batch_inference(
            BatchInferenceRequest(
                file_paths=(),
                sampling_rate_hz=12000,
                rpm=1500.0,
                model_bundle_path=bundle_path,
            )
        )
    except ValueError as exc:
        assert "at least one file path is required" in str(exc)
    else:
        raise AssertionError("expected empty batch to raise ValueError")
