from __future__ import annotations

from pathlib import Path

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
    assert sections["faults"] == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "机械松动",
        "轴承故障",
        "汽蚀",
    )
    assert "文件上传" in sections["pipeline"]
    assert sections["parameters"]["默认采样率"] == "12000 Hz"


def test_build_navigation_items_returns_chinese_titles_in_expected_order() -> None:
    items = build_navigation_items()

    assert [item["title"] for item in items] == [
        "系统首页",
        "单文件诊断",
        "批量诊断",
        "诊断报告",
    ]


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
        "device_id": "设备编号（选填）",
        "measurement_position": "测点位置（选填，例如：泵驱动端水平）",
        "export": "导出设置",
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
        {"文件名": "a.csv", "设备编号": "M2", "预测类别": "正常", "置信度": "0.910", "信号质量": "pass", "状态": "diagnosed"},
        {"文件名": "b.csv", "设备编号": "-", "预测类别": "-", "置信度": "-", "信号质量": "rejected", "状态": "rejected"},
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
    assert "spectrum unavailable" in view_data.visualization_availability.messages
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
