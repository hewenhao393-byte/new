from __future__ import annotations

from pathlib import Path

from pump_fault_app.services import AppBatchRunRequest, AppSingleRunRequest
from pump_fault_app.ui.streamlit_app import APP_SUBTITLE, APP_TITLE, build_home_sections
from pump_fault_app.ui.pages.batch_diagnosis import (
    build_batch_manifest_request,
    build_batch_table_rows,
    build_multi_file_batch_request,
)
from pump_fault_app.ui.pages.single_diagnosis import (
    build_probability_rows,
    build_single_run_request,
    build_single_summary_items,
)


def test_build_single_run_request_returns_service_request(tmp_path: Path) -> None:
    request = build_single_run_request(
        file_path=tmp_path / "record.csv",
        sampling_rate_hz=12000,
        rpm=1450.0,
        export_root=tmp_path / "exports",
    )

    assert isinstance(request, AppSingleRunRequest)
    assert request.sampling_rate_hz == 12000
    assert request.rpm == 1450.0
    assert request.export_root == tmp_path / "exports"


def test_build_probability_rows_returns_fixed_display_order() -> None:
    rows = build_probability_rows(
        (
            {"label": "松动", "probability": 0.11},
            {"label": "正常", "probability": 0.52},
            {"label": "汽蚀", "probability": 0.08},
        )
    )

    assert [row["label"] for row in rows] == [
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "机械松动",
        "轴承故障",
        "汽蚀",
    ]
    assert rows[0]["probability"] == 0.52
    assert rows[3]["probability"] == 0.11


def test_build_single_summary_items_contains_quality_window_and_runtime() -> None:
    items = build_single_summary_items(
        quality_text="pass",
        window_count=3,
        elapsed_seconds=0.234,
        message="诊断完成",
        warning_count=1,
    )

    assert items["信号质量"] == "pass"
    assert items["窗口数量"] == "3"
    assert items["推理耗时"] == "0.234 s"
    assert items["关键输出"] == "诊断完成"
    assert items["运行告警数"] == "1"


def test_build_batch_requests_return_service_request(tmp_path: Path) -> None:
    files_request = build_multi_file_batch_request(
        file_paths=(tmp_path / "a.csv", tmp_path / "b.csv"),
        sampling_rate_hz=12000,
        rpm=1450.0,
        export_root=tmp_path / "exports",
    )
    manifest_request = build_batch_manifest_request(
        manifest_path=tmp_path / "manifest.csv",
        export_root=tmp_path / "exports",
    )

    assert isinstance(files_request, AppBatchRunRequest)
    assert files_request.file_paths == (tmp_path / "a.csv", tmp_path / "b.csv")
    assert files_request.manifest_path is None
    assert isinstance(manifest_request, AppBatchRunRequest)
    assert manifest_request.manifest_path == tmp_path / "manifest.csv"
    assert manifest_request.file_paths is None


def test_build_batch_table_rows_returns_expected_columns() -> None:
    rows = build_batch_table_rows(
        [
            {"file_name": "a.csv", "diagnosis_label": "正常", "confidence": 0.91, "status": "diagnosed"},
            {"file_name": "b.csv", "diagnosis_label": None, "confidence": None, "status": "rejected"},
        ]
    )

    assert rows == [
        {"文件": "a.csv", "预测类别": "正常", "置信度": "0.910", "状态": "diagnosed"},
        {"文件": "b.csv", "预测类别": "-", "置信度": "-", "状态": "rejected"},
    ]


def test_build_home_sections_returns_title_and_modules() -> None:
    sections = build_home_sections()

    assert APP_TITLE in sections["title"]
    assert APP_SUBTITLE in sections["subtitle"]
    assert "单文件诊断" in sections["modules"]
    assert "批量诊断" in sections["modules"]


def test_ui_modules_only_depend_on_services_layer() -> None:
    root = Path("/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui")
    forbidden = (
        "pump_fault_app.io",
        "pump_fault_app.preprocessing",
        "pump_fault_app.feature_extraction",
        "pump_fault_app.prediction",
        "pump_fault_app.export.writer",
        "pump_fault_app.signal_reader",
    )
    required = "pump_fault_app.services"

    for path in [
        root / "streamlit_app.py",
        root / "pages" / "single_diagnosis.py",
        root / "pages" / "batch_diagnosis.py",
        root / "pages" / "report_view.py",
    ]:
        text = path.read_text(encoding="utf-8")
        if path.name != "streamlit_app.py":
            assert required in text
        for token in forbidden:
            assert token not in text
