from __future__ import annotations

from pathlib import Path

import numpy as np

from pump_fault_app.reporting import build_single_report_view_data
from pump_fault_app.services import AppSingleRunRequest, export_single_diagnosis_report, run_single_diagnosis
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_end_to_end_single_diagnosis_report_chain(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    signal_path = tmp_path / "record.csv"
    samples = (
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0)
        + 0.2 * np.sin(2.0 * np.pi * 50.0 * np.arange(4800) / 12000.0)
    )
    _write_signal_csv(signal_path, samples)

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            device_id="demo-pump",
            measurement_position="泵端水平",
        )
    )
    view_data = build_single_report_view_data(result)
    report_path = export_single_diagnosis_report(result, tmp_path / "acceptance_report.docx")

    assert result.summary.success is True
    assert result.visualization is not None
    assert result.visualization.time_domain is not None
    assert view_data.conclusion.diagnosis_label == "转子不平衡"
    assert report_path.exists()
    assert report_path.stat().st_size > 0
