from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pump_fault_app.batch import BatchInferenceRequest, run_batch_inference
from pump_fault_app.export import create_export_output_dir, export_batch_result
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_export_batch_result_writes_json_and_csv(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    _write_signal_csv(
        good_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    bad_path = tmp_path / "bad.csv"
    _write_signal_csv(bad_path, np.ones(1200, dtype=np.float64))

    batch_result = run_batch_inference(
        BatchInferenceRequest(
            file_paths=(good_path, bad_path),
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    export_result = export_batch_result(batch_result, output_dir=tmp_path / "exports")

    assert export_result.output_dir == tmp_path / "exports"
    assert export_result.json_path.exists()
    assert export_result.csv_path.exists()

    json_payload = json.loads(export_result.json_path.read_text(encoding="utf-8"))
    assert json_payload["total_count"] == 2
    assert json_payload["summaries"][0]["status"] == "diagnosed"
    assert json_payload["summaries"][1]["status"] == "rejected"

    with export_result.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "diagnosed"
    assert rows[0]["diagnosis_label"] == "转子不平衡"
    assert rows[1]["status"] == "rejected"
    assert rows[1]["rejection_reasons"] != ""


def test_export_batch_result_creates_output_directory(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    _write_signal_csv(
        good_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )

    batch_result = run_batch_inference(
        BatchInferenceRequest(
            file_paths=(good_path,),
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
        )
    )

    output_dir = tmp_path / "nested" / "exports"
    export_result = export_batch_result(batch_result, output_dir=output_dir)

    assert export_result.output_dir == output_dir
    assert output_dir.is_dir()


def test_create_export_output_dir_builds_timestamped_run_directory(tmp_path: Path) -> None:
    output_dir = create_export_output_dir(tmp_path, prefix="batch_run", timestamp="20260707_120000")

    assert output_dir == tmp_path / "batch_run_20260707_120000"
    assert output_dir.is_dir()
