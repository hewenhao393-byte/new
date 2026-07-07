from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pump_fault_app.app.cli import main
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_cli_main_prints_success_summary_json(tmp_path: Path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
        + 0.05 * np.random.default_rng(13).normal(size=4800)
    )
    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, samples)

    exit_code = main(
        [
            "--file",
            str(signal_path),
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
            "--device-id",
            "Motor-2",
            "--measurement-position",
            "泵端",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["success"] is True
    assert payload["status"] == "diagnosed"
    assert payload["diagnosis_label"] == "转子不平衡"
    assert payload["window_count"] == 3


def test_cli_main_returns_nonzero_for_quality_rejection(tmp_path: Path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "too_short.csv"
    _write_signal_csv(signal_path, np.ones(1200, dtype=np.float64))

    exit_code = main(
        [
            "--file",
            str(signal_path),
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["success"] is False
    assert payload["status"] == "rejected"
    assert payload["message"] == "信号质量不满足诊断条件"


def test_cli_main_parses_optional_signal_and_time_column(tmp_path: Path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0))

    exit_code = main(
        [
            "--file",
            str(signal_path),
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
            "--signal-column",
            "通道4",
            "--time-column",
            "time",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["success"] is True


def test_cli_main_can_export_single_result_files(tmp_path: Path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "record.csv"
    _write_signal_csv(signal_path, np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0))
    output_dir = tmp_path / "exports"

    exit_code = main(
        [
            "--file",
            str(signal_path),
            "--sampling-rate",
            "12000",
            "--rpm",
            "1500",
            "--model-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["success"] is True
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "diagnosis_summary.json").exists()
    assert (run_dirs[0] / "diagnosis_summary.csv").exists()
