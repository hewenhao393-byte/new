from __future__ import annotations

import pytest
pytest.skip("superseded by V3 UI tests during staged migration", allow_module_level=True)

from pathlib import Path
import subprocess
import sys

from pump_fault_app.domain.records import DiagnosisVisualizationData, SpectrumSeries, TimeDomainSeries, WaveletPacketEnergySeries
from pump_fault_app.reporting import build_single_report_view_data
from pump_fault_app.services import (
    AppBatchRunRequest,
    AppBatchRunResult,
    AppSingleRunRequest,
    AppSingleRunResult,
)
from pump_fault_app.ui.pages.report_view import (
    build_report_empty_message,
    build_report_overview,
)
from pump_fault_app.ui.streamlit_app import APP_SUBTITLE, APP_TITLE, build_home_sections, build_navigation_items
from pump_fault_app.ui.pages.batch_diagnosis import (
    build_batch_manifest_request,
    build_batch_table_rows,
    build_batch_filter_options,
    build_multi_file_batch_request,
)
from pump_fault_app.ui.pages.single_diagnosis import (
    build_probability_rows,
    build_single_run_request,
    build_single_summary_items,
    build_single_visual_availability,
    build_spectrum_rows,
    build_time_domain_rows,
    build_wavelet_packet_rows,
    get_single_advanced_field_labels,
)
from pump_fault_app.presentation.status import (
    build_engineering_risk_level,
    build_runtime_processing_status,
    build_user_runtime_notice,
)
from pump_fault_app.presentation.batch_diagnosis import (
    build_batch_table_rows as build_presentation_batch_table_rows,
    build_batch_task_statistics,
)
from pump_fault_app.presentation.report_view import build_diagnosis_highlight
from pump_fault_app.presentation.single_diagnosis import build_result_card_items, build_upload_signal_info
from pump_fault_app.presentation.system_overview import build_system_overview
from pump_fault_app.ui.branding import build_page_header, build_research_style


def test_build_single_run_request_returns_service_request(tmp_path: Path) -> None:
    request = build_single_run_request(
        file_path=tmp_path / "record.csv",
        sampling_rate_hz=12000,
        rpm=1450.0,
        vibration_direction="轴向",
        export_root=tmp_path / "exports",
        history_database_path=tmp_path / "history.sqlite3",
        history_report_dir=tmp_path / "history_reports",
    )

    assert isinstance(request, AppSingleRunRequest)
    assert request.sampling_rate_hz == 12000
    assert request.rpm == 1450.0
    assert request.vibration_direction == "轴向"
    assert request.export_root == tmp_path / "exports"
    assert request.history_database_path == tmp_path / "history.sqlite3"
    assert request.history_report_dir == tmp_path / "history_reports"


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


def test_build_single_summary_items_uses_neutral_processing_status_for_runtime_alerts() -> None:
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
    assert items["处理状态"] == "已完成数值稳定性保护处理，不影响诊断结果。"
    assert "运行告警数" not in items


def test_build_runtime_processing_status_is_neutral_but_preserves_detail_signal() -> None:
    assert build_runtime_processing_status(0) == "诊断流程正常完成。"
    assert build_runtime_processing_status(714) == "已完成数值稳定性保护处理，不影响诊断结果。"


def test_engineering_risk_level_follows_approved_display_rule() -> None:
    assert build_engineering_risk_level(success=True, label="正常") == "低风险"
    assert build_engineering_risk_level(success=True, label="汽蚀") == "需关注"
    assert build_engineering_risk_level(success=False, label=None) == "待复测"


def test_user_runtime_notice_hides_low_level_warning_text() -> None:
    assert build_user_runtime_notice(0) == ""
    assert build_user_runtime_notice(2) == "数值稳定性提示，不影响诊断结果。"


def test_report_view_uses_presentation_adapters_instead_of_page_helpers() -> None:
    report_view = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "pages" / "report_view.py"
    text = report_view.read_text(encoding="utf-8")

    assert "pump_fault_app.presentation.single_diagnosis" in text
    assert "from pump_fault_app.ui.pages.single_diagnosis import" not in text


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
        {
            "文件名": "a.csv",
            "设备编号": "-",
            "预测类别": "正常",
            "置信度": "91.0%",
            "信号质量": "pass",
            "状态": "已完成",
        },
        {
            "文件名": "b.csv",
            "设备编号": "-",
            "预测类别": "-",
            "置信度": "-",
            "信号质量": "rejected",
            "状态": "质量拒绝",
        },
    ]


def test_build_batch_task_statistics_summarizes_existing_diagnosis_results() -> None:
    statistics = build_batch_task_statistics(
        [
            {"status": "diagnosed", "diagnosis_label": "正常", "confidence": 0.9},
            {"status": "diagnosed", "diagnosis_label": "汽蚀", "confidence": 0.98},
            {"status": "rejected", "diagnosis_label": None, "confidence": None},
        ]
    )

    assert statistics == {
        "总文件数": "3",
        "完成数量": "2",
        "异常数量": "1",
        "平均置信度": "94.0%",
    }


def test_build_upload_signal_info_uses_only_uploaded_metadata_and_user_inputs() -> None:
    rows = build_upload_signal_info(
        file_name="demo.csv",
        file_size_bytes=2048,
        sampling_rate_hz=20000,
        rpm=2070.0,
        signal_column="0",
        time_column="time",
        measurement_position="泵驱端水平",
    )

    assert rows == [
        {"项目": "文件名称", "内容": "demo.csv"},
        {"项目": "文件大小", "内容": "2.0 KB"},
        {"项目": "输入采样率", "内容": "20000 Hz"},
        {"项目": "转速", "内容": "2070.0 rpm"},
        {"项目": "信号列", "内容": "0"},
        {"项目": "时间列", "内容": "time"},
        {"项目": "测点位置", "内容": "泵驱端水平"},
    ]


def test_build_diagnosis_highlight_uses_normal_fault_and_warning_display_tones() -> None:
    assert build_diagnosis_highlight("正常", "93.2%", "可信诊断") == {
        "tone": "normal",
        "status": "正常",
        "label": "正常",
        "confidence": "93.2%",
        "grade": "可信诊断",
    }
    assert build_diagnosis_highlight("汽蚀", "99.9%", "可信诊断")["tone"] == "fault"
    assert build_diagnosis_highlight("-", "-", "建议复测")["tone"] == "warning"


def test_build_home_sections_returns_title_and_modules() -> None:
    sections = build_home_sections()

    assert APP_TITLE in sections["title"]
    assert APP_SUBTITLE in sections["subtitle"]
    assert sections["faults"] == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "机械松动",
        "轴承故障",
        "汽蚀",
    )
    assert "信号采集" in sections["pipeline"]
    assert "特征分析" in sections["pipeline"]
    assert "BP智能识别" in sections["pipeline"]
    assert sections["parameters"]["默认采样率"] == "12000 Hz"
    assert sections["fault_cards"][0]["title"] == "正常状态"
    assert sections["pipeline"][-1] == "自动报告生成"


def test_system_overview_is_built_from_the_formal_contract() -> None:
    overview = build_system_overview()

    assert overview["title"] == "系统原理与模型说明"
    assert overview["subtitle"] == "基于振动机理特征与BP神经网络的水泵六分类故障诊断方法"
    assert overview["pipeline"] == (
        "振动信号",
        "预处理",
        "滑动窗口",
        "21维特征",
        "BP模型",
        "概率融合",
        "六类故障",
    )
    assert len(overview["features"]) == 21
    assert [group["title"] for group in overview["feature_groups"]] == [
        "时域冲击特征",
        "转频及倍频特征",
        "频域能量特征",
        "包络与小波特征",
    ]
    assert [len(group["features"]) for group in overview["feature_groups"]] == [6, 3, 2, 10]
    assert sum(len(group["features"]) for group in overview["feature_groups"]) == 21
    assert tuple(
        feature_name
        for group in overview["feature_groups"]
        for feature_name in group["feature_names"]
    ) == overview["formal_feature_names"]
    assert all(group["explanation"] for group in overview["feature_groups"])
    assert overview["model"]["名称"] == "BP神经网络六分类模型"
    assert overview["labels"] == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "机械松动",
        "轴承故障",
        "汽蚀",
    )
    assert overview["parameters"]["统一采样率"] == "12000 Hz"
    assert overview["parameters"]["分析频带"] == "10～5000 Hz"
    assert overview["parameters"]["窗口与步长"] == "2400点 / 1200点"
    assert overview["parameters"]["小波包"] == "db6，3层分解"


def test_system_overview_explains_window_prediction_and_probability_fusion() -> None:
    overview = build_system_overview()

    assert overview["model"]["诊断方式"] == "窗口级预测 + 多窗口概率融合"
    assert "0.2 s" in overview["model"]["窗口级预测"]
    assert "六类状态概率" in overview["model"]["窗口级预测"]
    assert "同类概率取平均" in overview["model"]["多窗口概率融合"]
    assert "最高平均概率" in overview["model"]["多窗口概率融合"]


def test_system_overview_fault_feature_rows_match_the_frozen_feature_scope() -> None:
    overview = build_system_overview()

    assert [row["故障"] for row in overview["fault_feature_rows"]] == [
        "转子不平衡",
        "联轴器不对中",
        "机械松动",
        "轴承故障",
        "汽蚀",
    ]
    table_text = " ".join(
        str(value)
        for row in overview["fault_feature_rows"]
        for value in row.values()
    )
    assert "1X绝对幅值" not in table_text
    assert "2X/1X" in table_text
    assert "包络峭度" in table_text
    assert "频谱熵" in table_text


def test_system_overview_parameter_rows_are_wide_enough_for_full_values() -> None:
    overview = build_system_overview()

    assert [len(row) for row in overview["parameter_rows"]] == [3, 2]
    values = [item["value"] for row in overview["parameter_rows"] for item in row]
    assert "10～5000 Hz" in values
    assert "2400点 / 1200点" in values


def test_system_overview_page_renders_explanations_without_diagnosis_logic() -> None:
    page = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "pages" / "system_overview.py"
    text = page.read_text(encoding="utf-8")

    assert 'overview["feature_groups"]' in text
    assert 'overview["fault_feature_rows"]' in text
    assert 'overview["parameter_rows"]' in text
    assert "st.metric" not in text
    assert "run_formal_inference" not in text
    assert "fuse_window_predictions" not in text
    assert "extract_features" not in text


def test_research_style_uses_light_scientific_palette_without_model_claims() -> None:
    style = build_research_style()

    assert "#f7f9fc" in style
    assert "#ffffff" in style
    assert ".research-card" in style
    assert "#111418" not in style
    assert "XGBoost" not in style
    assert "time.sleep" not in style


def test_page_header_omits_internal_navigation_numbering() -> None:
    header = build_page_header(title="单文件智能诊断", subtitle="真实振动信号六分类诊断。")

    assert "单文件智能诊断" in header
    assert "真实振动信号六分类诊断。" in header
    assert "01 /" not in header
    assert "console-page-header" not in header


def test_build_navigation_items_returns_chinese_titles_in_expected_order() -> None:
    items = build_navigation_items()

    assert [item["title"] for item in items] == [
        "系统首页",
        "单文件诊断",
        "批量诊断",
        "诊断结果",
        "历史记录",
        "模型持续优化",
        "系统说明",
    ]
    assert [item["nav_label"] for item in items] == [
        "01 系统首页",
        "02 单文件诊断",
        "03 批量诊断",
        "04 诊断结果",
        "05 历史记录",
        "06 模型优化",
        "07 系统说明",
    ]


def test_history_page_uses_history_repository_without_diagnosis_logic() -> None:
    history_page = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "history.py"
    )
    text = history_page.read_text(encoding="utf-8")

    assert "历史诊断记录" in text
    assert '"下载"' in text
    assert "SQLiteDiagnosisHistoryRepository" in text
    assert "pump_fault_app.presentation.history" in text
    assert "run_formal_inference" not in text
    assert "extract_formal_features" not in text
    assert "fuse_window_predictions" not in text
    assert "load_formal_bp_bundle" not in text


def test_history_page_requires_confirmation_before_deletion() -> None:
    history_page = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "history.py"
    )
    text = history_page.read_text(encoding="utf-8")

    assert '"删除"' in text
    assert '"下载"' in text
    assert "确认删除" in text
    assert "取消" in text
    assert "delete_history_record" in text
    assert "st.session_state" in text
    assert "report_delete_failed" in text
    assert "### 删除记录" not in text
    assert "操作" in text
    assert "for record in records" in text
    assert "history-download-" in text
    assert "history-delete-start-" in text
    assert "white-space: nowrap" in text
    assert "text-overflow: ellipsis" in text
    assert "escape(record.file_name)" in text
    assert "{record.sampling_rate_hz} Hz · {record.rpm:g} rpm" in text
    assert "run_formal_inference" not in text
    assert "extract_formal_features" not in text
    assert "fuse_window_predictions" not in text
    assert "load_formal_bp_bundle" not in text


def test_navigation_includes_model_optimization_without_formal_inference_changes() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    page_source = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "model_optimization.py"
    ).read_text(encoding="utf-8")

    assert "模型持续优化" in app_source
    assert "候选模型" in page_source
    assert "人工批准" in page_source
    assert "21维" in page_source
    assert "run_formal_inference" not in page_source


def test_model_optimization_page_displays_one_selectable_history_sample_at_a_time() -> None:
    page_source = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "model_optimization.py"
    ).read_text(encoding="utf-8")

    assert "选择需要人工确认的历史样本" in page_source
    assert "当前第 {selected_index + 1} / {len(records)} 条" in page_source
    assert "for record in records:" not in page_source


def test_model_optimization_page_uses_full_text_cards_for_formal_model_details() -> None:
    page_source = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "model_optimization.py"
    ).read_text(encoding="utf-8")

    assert "model-version-card" in page_source
    assert "overflow-wrap: anywhere" in page_source
    assert "escape(formal.version)" in page_source
    assert "escape(formal.status)" in page_source


def test_single_diagnosis_page_shows_only_friendly_history_warning() -> None:
    single_page = (
        Path(__file__).resolve().parents[1]
        / "pump_fault_app"
        / "ui"
        / "pages"
        / "single_diagnosis.py"
    )
    text = single_page.read_text(encoding="utf-8")

    assert "result.history_warning" in text
    assert "诊断已完成，但历史记录或Word报告未能保存。" in text
    assert "sqlite3" not in text


def test_streamlit_app_imports_without_circular_page_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", "import pump_fault_app.ui.streamlit_app"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_batch_display_helpers_are_owned_by_presentation_layer() -> None:
    assert build_presentation_batch_table_rows(
        [{"file_name": "a.csv", "status": "diagnosed", "diagnosis_label": "正常", "confidence": 0.9}]
    )[0]["状态"] == "已完成"

    report_view = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "pages" / "report_view.py"
    text = report_view.read_text(encoding="utf-8")
    assert "pump_fault_app.presentation.batch_diagnosis" in text
    assert "from pump_fault_app.ui.pages.batch_diagnosis import build_batch_table_rows" not in text


def test_home_page_cta_switches_to_the_registered_navigation_page() -> None:
    app_path = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "streamlit_app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "st.switch_page(single_page)" in text
    assert "st.switch_page(\"pump_fault_app/ui/pages/single_diagnosis.py\")" not in text


def test_top_navigation_is_rendered_outside_the_collapsible_sidebar() -> None:
    app_path = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui" / "streamlit_app.py"
    text = app_path.read_text(encoding="utf-8")

    assert "def render_top_navigation" in text
    assert "render_top_navigation(st, page_by_path)" in text
    assert "key=f'top-nav-" in text


def test_ui_keeps_real_service_contracts_and_avoids_demo_sensor_inputs() -> None:
    root = Path(__file__).resolve().parents[1] / "pump_fault_app" / "ui"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert "run_single_diagnosis" in text
    assert "run_batch_diagnosis" in text
    assert "XGBoost" not in text
    assert "轴承温度" not in text
    assert "流量 (m³/h)" not in text
    assert "time.sleep" not in text


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


def test_get_single_advanced_field_labels_returns_user_friendly_labels() -> None:
    labels = get_single_advanced_field_labels()

    assert labels == {
        "signal_column": "振动信号列（选填，留空时自动识别）",
        "time_column": "时间列（选填，留空时自动识别）",
        "export": "导出设置",
    }


def test_build_result_card_items_reports_consistency_and_risk_without_changing_prediction() -> None:
    result = AppSingleRunResult(
        inference_result=type(
            "Inference",
            (),
            {
                "window_predictions": (
                    type("Prediction", (), {"predicted_label": "汽蚀"})(),
                    type("Prediction", (), {"predicted_label": "汽蚀"})(),
                    type("Prediction", (), {"predicted_label": "正常"})(),
                )
            },
        )(),
        summary=type(
            "Summary",
            (),
            {"success": True, "diagnosis_label": "汽蚀", "confidence": 0.92},
        )(),
        export_result=None,
        visualization=None,
    )

    assert build_result_card_items(result) == {
        "诊断结果": "汽蚀",
        "置信度": "92.0%",
        "窗口一致率": "66.7%",
        "风险等级": "需关注",
    }


def test_build_single_visual_availability_hides_missing_service_backed_charts() -> None:
    result = AppSingleRunResult(
        inference_result=type(
            "InferenceResult",
            (),
            {
                "window_predictions": (object(),),
            },
        )(),
        summary=type(
            "Summary",
            (),
            {
                "top_probabilities": ({"label": "正常", "probability": 0.9},),
            },
        )(),
        export_result=None,
        visualization=DiagnosisVisualizationData(
            time_domain=TimeDomainSeries(
                time_s=(0.0, 0.1),
                amplitude=(0.1, -0.1),
                sampling_rate_hz=12000,
                point_count=2,
                downsampled_for_display=False,
            ),
            frequency_spectrum=SpectrumSeries(
                frequency_hz=(0.0, 10.0),
                amplitude=(0.0, 1.0),
                frequency_min_hz=0.0,
                frequency_max_hz=10.0,
                resolution_hz=10.0,
            ),
            envelope_spectrum=SpectrumSeries(
                frequency_hz=(0.0, 10.0),
                amplitude=(0.0, 0.5),
                frequency_min_hz=0.0,
                frequency_max_hz=10.0,
                resolution_hz=10.0,
            ),
            wavelet_packet_energy=WaveletPacketEnergySeries(
                band_labels=("0-750 Hz",),
                band_start_hz=(0.0,),
                band_end_hz=(750.0,),
                energy_ratio=(1.0,),
                wavelet="db6",
                decomposition_level=3,
            ),
        ),
    )

    availability = build_single_visual_availability(result)

    assert availability["probability_chart"] is True
    assert availability["window_distribution"] is True
    assert availability["waveform"] is True
    assert availability["spectrum"] is True
    assert availability["envelope"] is True
    assert availability["wavelet"] is True


def test_build_visualization_rows_return_expected_columns() -> None:
    time_rows = build_time_domain_rows(
        TimeDomainSeries(
            time_s=(0.0, 0.1),
            amplitude=(1.0, -1.0),
            sampling_rate_hz=12000,
            point_count=2,
            downsampled_for_display=False,
        )
    )
    spectrum_rows = build_spectrum_rows(
        SpectrumSeries(
            frequency_hz=(0.0, 25.0),
            amplitude=(0.0, 0.8),
            frequency_min_hz=0.0,
            frequency_max_hz=25.0,
            resolution_hz=25.0,
        )
    )
    wavelet_rows = build_wavelet_packet_rows(
        WaveletPacketEnergySeries(
            band_labels=("0-750 Hz", "750-1500 Hz"),
            band_start_hz=(0.0, 750.0),
            band_end_hz=(750.0, 1500.0),
            energy_ratio=(0.7, 0.3),
            wavelet="db6",
            decomposition_level=3,
        )
    )

    assert list(time_rows.columns) == ["time_seconds", "amplitude"]
    assert list(spectrum_rows.columns) == ["frequency_hz", "amplitude"]
    assert wavelet_rows == [
        {"band_label": "0-750 Hz", "energy_ratio": 0.7},
        {"band_label": "750-1500 Hz", "energy_ratio": 0.3},
    ]


def test_build_batch_filter_options_includes_all_and_predicted_labels() -> None:
    options = build_batch_filter_options(
        [
            {"diagnosis_label": "正常"},
            {"diagnosis_label": "轴承故障"},
            {"diagnosis_label": "正常"},
            {"diagnosis_label": None},
        ]
    )

    assert options == ["全部", "正常", "轴承故障"]


def test_build_batch_table_rows_returns_expected_columns() -> None:
    rows = build_batch_table_rows(
        [
            {
                "file_name": "a.csv",
                "device_id": "M2",
                "diagnosis_label": "正常",
                "confidence": 0.91,
                "status": "diagnosed",
                "signal_quality": "pass",
            },
            {
                "file_name": "b.csv",
                "device_id": None,
                "diagnosis_label": None,
                "confidence": None,
                "status": "rejected",
                "signal_quality": "rejected",
            },
        ]
    )

    assert rows == [
        {"文件名": "a.csv", "设备编号": "M2", "预测类别": "正常", "置信度": "91.0%", "信号质量": "pass", "状态": "已完成"},
        {"文件名": "b.csv", "设备编号": "-", "预测类别": "-", "置信度": "-", "信号质量": "rejected", "状态": "质量拒绝"},
    ]


def test_build_report_empty_message_matches_requirement() -> None:
    assert build_report_empty_message() == "暂无诊断结果，请先完成单文件诊断或批量诊断。"


def test_build_report_overview_prefers_latest_kind() -> None:
    single = AppSingleRunResult(inference_result=object(), summary=object(), export_result=None, visualization=None)
    batch = AppBatchRunResult(batch_result=object(), export_result=None)

    overview = build_report_overview(single_result=single, batch_result=batch, latest_kind="batch")

    assert overview["active_kind"] == "batch"


def test_report_view_uses_visualization_contract_for_chart_availability() -> None:
    result = AppSingleRunResult(
        inference_result=type("InferenceResult", (), {"raw_signal": object()})(),
        summary=object(),
        export_result=None,
        visualization=DiagnosisVisualizationData(
            time_domain=TimeDomainSeries(
                time_s=(0.0, 0.1),
                amplitude=(0.2, -0.2),
                sampling_rate_hz=12000,
                point_count=2,
                downsampled_for_display=False,
            ),
            frequency_spectrum=SpectrumSeries(
                frequency_hz=(0.0, 50.0),
                amplitude=(0.0, 1.0),
                frequency_min_hz=0.0,
                frequency_max_hz=50.0,
                resolution_hz=50.0,
            ),
            envelope_spectrum=SpectrumSeries(
                frequency_hz=(0.0, 20.0),
                amplitude=(0.0, 0.8),
                frequency_min_hz=0.0,
                frequency_max_hz=20.0,
                resolution_hz=20.0,
            ),
            wavelet_packet_energy=WaveletPacketEnergySeries(
                band_labels=("0-750 Hz",),
                band_start_hz=(0.0,),
                band_end_hz=(750.0,),
                energy_ratio=(1.0,),
                wavelet="db6",
                decomposition_level=3,
            ),
        ),
    )

    view_data = build_single_report_view_data(result)

    assert view_data.visualization_availability.time_domain is True
    assert view_data.visualization_availability.frequency_spectrum is True
    assert view_data.visualization_availability.envelope_spectrum is True
    assert view_data.visualization_availability.wavelet_packet_energy is True
    assert view_data.time_domain is not None
    assert view_data.frequency_spectrum is not None
    assert view_data.envelope_spectrum is not None
    assert view_data.wavelet_packet_energy is not None


def test_report_view_handles_missing_visualization_without_crashing() -> None:
    result = AppSingleRunResult(
        inference_result=object(),
        summary=object(),
        export_result=None,
        visualization=None,
    )

    view_data = build_single_report_view_data(result)

    assert view_data.visualization_availability.time_domain is False
    assert view_data.visualization_availability.frequency_spectrum is False
    assert view_data.visualization_availability.envelope_spectrum is False
    assert view_data.visualization_availability.wavelet_packet_energy is False
    assert "暂无可视化数据" in view_data.visualization_availability.messages[0]


def test_report_view_handles_partial_visualization_and_preserves_other_sections() -> None:
    result = AppSingleRunResult(
        inference_result=object(),
        summary=object(),
        export_result=None,
        visualization=DiagnosisVisualizationData(
            time_domain=TimeDomainSeries(
                time_s=(0.0, 0.1),
                amplitude=(0.2, -0.2),
                sampling_rate_hz=12000,
                point_count=2,
                downsampled_for_display=False,
            ),
            frequency_spectrum=None,
            envelope_spectrum=SpectrumSeries(
                frequency_hz=(0.0, 20.0),
                amplitude=(0.0, 0.8),
                frequency_min_hz=0.0,
                frequency_max_hz=20.0,
                resolution_hz=20.0,
            ),
            wavelet_packet_energy=None,
            warnings=("spectrum unavailable",),
        ),
    )

    view_data = build_single_report_view_data(result)

    assert view_data.visualization_availability.time_domain is True
    assert view_data.visualization_availability.frequency_spectrum is False
    assert view_data.visualization_availability.envelope_spectrum is True
    assert view_data.visualization_availability.wavelet_packet_energy is False
    assert "频谱图数据不可用" in view_data.visualization_availability.messages
    assert "小波包能量图数据不可用" in view_data.visualization_availability.messages
    assert "部分振动特征图未生成，正式诊断结果不受影响。" in view_data.visualization_availability.messages
    assert "spectrum unavailable" not in view_data.visualization_availability.messages
    assert view_data.time_domain is not None
    assert view_data.envelope_spectrum is not None


def test_report_view_module_uses_report_view_data_builder() -> None:
    path = Path("/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/pages/report_view.py")
    text = path.read_text(encoding="utf-8")

    assert "build_single_report_view_data" in text
    assert "build_report_visual_availability" not in text


def test_report_view_module_uses_word_export_entrypoint() -> None:
    path = Path("/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/pages/report_view.py")
    text = path.read_text(encoding="utf-8")

    assert "export_single_diagnosis_report" in text
    assert "export_single_report_to_docx" not in text
    assert "导出 Word 报告" in text
