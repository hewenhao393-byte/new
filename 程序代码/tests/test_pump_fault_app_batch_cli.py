from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pump_fault_app.app.batch_cli import main
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_batch_cli_main_prints_batch_summary_json(tmp_path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    _write_signal_csv(
        good_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    bad_path = tmp_path / "bad.csv"
    _write_signal_csv(bad_path, np.ones(1200, dtype=np.float64))

    exit_code = main(
        [
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
            "--files",
            str(good_path),
            str(bad_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["total_count"] == 2
    assert payload["diagnosed_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["summaries"][0]["status"] == "diagnosed"
    assert payload["summaries"][1]["status"] == "rejected"


def test_batch_cli_main_can_export_files(tmp_path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    good_path = tmp_path / "good.csv"
    _write_signal_csv(
        good_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    output_dir = tmp_path / "exports"

    exit_code = main(
        [
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--files",
            str(good_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["total_count"] == 1
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "batch_diagnosis_summary.json").exists()
    assert (run_dirs[0] / "batch_diagnosis_summary.csv").exists()


def test_batch_cli_main_can_run_from_manifest(tmp_path, capsys) -> None:
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
            {"file_path": str(good_path), "sampling_rate_hz": 12000, "rpm": 1500.0},
            {"file_path": str(bad_path), "sampling_rate_hz": 12000, "rpm": 1500.0},
        ]
    ).to_csv(manifest_path, index=False)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--model-bundle",
            str(bundle_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["total_count"] == 2
    assert payload["diagnosed_count"] == 1
    assert payload["rejected_count"] == 1
