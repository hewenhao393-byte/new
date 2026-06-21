from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import PipelineConfig
from pump_diagnosis.metadata import ConditionCatalog, build_file_index, split_file_index


FAULTS = {
    "正常": "正常状态1",
    "转子不平衡": "泵不平衡1",
    "联轴器不对中": "角向不对中1",
    "松动": "软脚1",
    "轴承故障": "轴承外圈故障1",
    "汽蚀": "出口汽蚀1",
}


def _write_trace_csv(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": [0.00005, 0.00010, 0.00015, 0.00020],
            "0": [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset],
            "1": [0.4 + offset, 0.3 + offset, 0.2 + offset, 0.1 + offset],
        }
    ).to_csv(path, index=False)


def test_file_index_is_one_row_per_run_and_split_has_no_file_leakage(tmp_path: Path) -> None:
    root = tmp_path / "Vibration"
    rpm_lookup: dict[tuple[str, int, str], float] = {}
    for label_index, (label, fault) in enumerate(FAULTS.items()):
        for severity in range(5):
            machine_id = f"Motor-{severity + 1}"
            raw_fault = fault if label == "正常" else fault[:-1] + str(severity + 1)
            path = (
                root
                / machine_id
                / "70"
                / raw_fault
                / f"振动_{machine_id}_70_时域-{raw_fault}-通道4.csv"
            )
            _write_trace_csv(path, offset=(label_index * 10 + severity) / 100)
            rpm_lookup[(machine_id, 70, raw_fault)] = 2070.0 + severity

    config = PipelineConfig(data_root=root, output_root=tmp_path / "output")
    index = build_file_index(config, rpm_lookup=rpm_lookup)

    assert len(index) == 60
    assert set(index["label"]) == set(FAULTS)
    assert set(index["channel"]) == {4}
    assert set(index["original_fs"]) == {20_000}
    assert index["run_id"].is_unique
    assert index.groupby("file_path").size().eq(2).all()

    train, test, report = split_file_index(index, config)

    assert set(train["label"]) == set(FAULTS)
    assert set(test["label"]) == set(FAULTS)
    assert set(train["file_path"]).isdisjoint(test["file_path"])
    assert set(train["run_id"]).isdisjoint(test["run_id"])
    assert report["source_file_overlap_count"] == 0
    assert report["hash_overlap_count"] == 0
    assert train["run_id"].is_unique
    assert test["run_id"].is_unique


def test_file_index_excludes_non_six_class_and_other_channels(tmp_path: Path) -> None:
    root = tmp_path / "Vibration"
    included = root / "Motor-4" / "70" / "正常状态1" / "正常状态1-通道4.csv"
    noisy = root / "Motor-4" / "70" / "正常加噪声" / "正常加噪声-通道4.csv"
    channel3 = root / "Motor-4" / "70" / "正常状态1" / "正常状态1-通道3.csv"
    for path in (included, noisy, channel3):
        _write_trace_csv(path)

    config = PipelineConfig(data_root=root, output_root=tmp_path / "output")
    index = build_file_index(
        config,
        rpm_lookup={("Motor-4", 70, "正常状态1"): 2070.0},
    )

    assert index["file_path"].nunique() == 1
    assert set(index["raw_fault"]) == {"正常状态1"}


def test_condition_catalog_maps_chinese_faults_to_actual_rpm(tmp_path: Path) -> None:
    workbook = tmp_path / "conditions.xlsx"
    rows = [
        ["Measurements"],
        ["Order", "Setup", "Failure description", "Severity", "Speed (%)", "Speed (RPM)"],
        [1, "motor 4", "Healthy 1", "N/A", 0.7, 2070],
        [2, "motor 4", "Cavitation suction", 4, 0.7, 2085],
        [3, "motor 4", "Cavitation discharge", 3, 0.7, 2080],
        [4, "motor 2", "Bearing BPFI", 2, 0.75, 1110],
    ]
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Ordered Measurements", index=False, header=False)

    catalog = ConditionCatalog.from_workbook(workbook)

    assert catalog.rpm_for("Motor-4", 70, "正常状态1") == 2070
    assert catalog.rpm_for("Motor-4", 70, "吸入口汽蚀4") == 2085
    assert catalog.rpm_for("Motor-4", 70, "出口汽蚀3") == 2080
    assert catalog.rpm_for("Motor-2", 75, "轴承内圈故障2") == 1110


def test_condition_catalog_rejects_conflicting_rpm_rows(tmp_path: Path) -> None:
    workbook = tmp_path / "conflicting_conditions.xlsx"
    rows = [
        ["Measurements"],
        ["Order", "Setup", "Failure description", "Severity", "Speed (%)", "Speed (RPM)"],
        [1, "motor 4", "Healthy 1", "N/A", 0.7, 2070],
        [2, "motor 4", "Healthy 1", "N/A", 0.7, 2080],
    ]
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Ordered Measurements", index=False, header=False)

    with pytest.raises(ValueError, match="冲突转速"):
        ConditionCatalog.from_workbook(workbook)
