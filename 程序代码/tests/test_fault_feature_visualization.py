from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.fault_feature_visualization import (
    FEATURE_COLUMNS,
    FigureManifestEntry,
    VisualizationConfig,
    build_figure_manifest_frame,
    select_representative_records,
)


def test_select_representative_records_picks_closest_group_to_class_center() -> None:
    rows = []
    for label, base_values in {
        "正常": [0.0, 1.0, 2.0],
        "松动": [10.0, 11.0, 12.0],
    }.items():
        for record_index, feature_value in enumerate(base_values):
            for window_index in range(2):
                row = {
                    "source_file": f"/tmp/{label}_{record_index}.csv",
                    "window_id": f"{label}_{record_index}_{window_index}",
                    "window_start": window_index * 1200,
                    "window_end": window_index * 1200 + 2400,
                    "label": label,
                    "record_index": record_index,
                    "group_id": f"/tmp/{label}_{record_index}.csv::record_{record_index}",
                }
                for feature in FEATURE_COLUMNS:
                    row[feature] = feature_value if feature == FEATURE_COLUMNS[0] else 0.0
                rows.append(row)
    frame = pd.DataFrame(rows)

    selected = select_representative_records(frame, FEATURE_COLUMNS)

    assert selected["正常"]["record_index"] == 1
    assert selected["松动"]["record_index"] == 1


def test_build_figure_manifest_frame_records_paths_and_parameters(tmp_path: Path) -> None:
    config = VisualizationConfig(output_root=tmp_path)
    entry = FigureManifestEntry(
        figure_id="motor2_normal_time",
        device="Motor-2",
        fault_label="正常",
        figure_type="time_waveform",
        source_file="/tmp/motor2_normal.csv",
        record_index=3,
        time_segment="0.0-1.0 s",
        processing_params="fs=12000; bandpass=5-5000 Hz",
        png_path=tmp_path / "a.png",
        pdf_path=tmp_path / "a.pdf",
        csv_path=tmp_path / "a.csv",
    )

    frame = build_figure_manifest_frame([entry], config)

    assert frame.loc[0, "figure_id"] == "motor2_normal_time"
    assert frame.loc[0, "device"] == "Motor-2"
    assert frame.loc[0, "fault_label"] == "正常"
    assert frame.loc[0, "processing_params"] == "fs=12000; bandpass=5-5000 Hz"
    assert frame.loc[0, "png_path"].endswith("a.png")
