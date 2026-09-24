from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest
from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.domain.labels import display_label
from pump_fault_app.history import V3HistoryRepository
from pump_fault_app.history.service import DEFAULT_HISTORY_DATABASE_PATH
from pump_fault_app.reporting import build_single_report_view_data
from pump_fault_app.services import (
    AppBatchRunRequest,
    AppSingleRunRequest,
    export_single_diagnosis_report,
    run_batch_diagnosis,
    run_single_diagnosis,
)
from pump_fault_app.ui.branding import build_page_header


def build_multichannel_run_request(
    *,
    channels: tuple[ChannelInput, ...],
    sampling_rate_hz: int,
    rpm: float,
    model_directory: Path | None = None,
    vibration_direction: str | None = None,
    export_root: Path | None = None,
    history_database_path: Path | None = None,
    history_report_dir: Path | None = None,
) -> AppSingleRunRequest:
    if not channels:
        raise ValueError("至少需要一个通道文件")
    time_presence = tuple(item.time_column is not None for item in channels)
    if any(time_presence) and not all(time_presence):
        raise ValueError("时间列必须全部提供或全部不提供")
    return AppSingleRunRequest(
        MultiChannelInferenceRequest(channels, sampling_rate_hz, rpm, model_directory),
        vibration_direction=vibration_direction,
        export_root=export_root,
        history_database_path=history_database_path,
        history_report_dir=history_report_dir,
    )


def _st():
    import streamlit as st

    return st


def _persist(uploaded: Any, channel: str) -> Path:
    directory = Path(tempfile.mkdtemp(prefix=f"pump_v3_{channel.lower()}_"))
    target = directory / Path(uploaded.name).name
    target.write_bytes(uploaded.getbuffer())
    return target


def _probability_rows(probabilities: tuple[float, ...] | None) -> list[dict[str, Any]]:
    if probabilities is None:
        return []
    return [
        {"类别": display_label(label), "模型输出概率": float(value)}
        for label, value in zip(FORMAL_LABEL_ORDER, probabilities)
    ]


def _render_result(st: Any, app_result: Any) -> None:
    result = app_result.inference_result
    st.markdown("## 多通道融合结果")
    if result.status == "failed":
        st.error("所有输入通道均无效，本次诊断失败。")
    else:
        st.success(f"最终诊断：{display_label(result.predicted_label)}")
        columns = st.columns(3)
        columns[0].metric("有效通道", "、".join(result.valid_channels))
        columns[1].metric("通道一致性", result.agreement_level or "单通道")
        columns[2].metric("推理契约", result.contract_version)
        st.markdown("### 六类模型输出概率")
        st.dataframe(pd.DataFrame(_probability_rows(result.fused_probabilities)), hide_index=True, use_container_width=True)

    st.markdown("## 各通道诊断结果")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "通道": item.channel,
                    "状态": item.status,
                    "诊断类别": display_label(item.predicted_label) if item.predicted_label else "-",
                    "窗口数": item.window_count,
                    "窗口一致率": "-" if item.window_consistency is None else f"{item.window_consistency:.1%}",
                    "无效原因": item.failure_message or "-",
                }
                for item in result.channel_results
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    selected = st.selectbox("选择通道查看详情", result.input_channels)
    channel_result = next(item for item in result.channel_results if item.channel == selected)
    st.markdown(f"### {selected} 模型输出概率")
    st.dataframe(pd.DataFrame(_probability_rows(channel_result.class_probabilities)), hide_index=True, use_container_width=True)
    tabs = st.tabs(("时域波形", "频谱", "包络谱", "小波包能量"))
    visualization = channel_result.visualization
    if visualization is None:
        for tab in tabs:
            with tab:
                st.info(f"{selected} 无可用图表数据。")
    else:
        with tabs[0]:
            series = visualization.time_domain
            if series is None:
                st.info("无可用时域数据。")
            else:
                frame = pd.DataFrame({"时间 / s": series.time_s, "幅值": series.amplitude}).set_index("时间 / s")
                st.line_chart(frame)
        with tabs[1]:
            series = visualization.frequency_spectrum
            if series is None:
                st.info("无可用频谱数据。")
            else:
                frame = pd.DataFrame({"频率 / Hz": series.frequency_hz, "幅值": series.amplitude}).set_index("频率 / Hz")
                st.line_chart(frame)
        with tabs[2]:
            series = visualization.envelope_spectrum
            if series is None:
                st.info("无可用包络谱数据。")
            else:
                frame = pd.DataFrame({"频率 / Hz": series.frequency_hz, "幅值": series.amplitude}).set_index("频率 / Hz")
                st.line_chart(frame)
        with tabs[3]:
            series = visualization.wavelet_packet_energy
            if series is None:
                st.info("无可用小波包能量数据。")
            else:
                frame = pd.DataFrame({"频带": series.band_labels, "能量占比": series.energy_ratio}).set_index("频带")
                st.bar_chart(frame)

    if app_result.report_path and Path(app_result.report_path).is_file():
        st.download_button(
            "下载Word报告",
            data=Path(app_result.report_path).read_bytes(),
            file_name=Path(app_result.report_path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def single_diagnosis_page() -> None:
    st = _st()
    st.markdown(
        build_page_header(
            title="CatBoost43 多通道诊断",
            subtitle="CH3、CH4、CH5 分别上传文件，各通道独立诊断后等权概率融合。",
        ),
        unsafe_allow_html=True,
    )
    uploads = {}
    for channel in ("CH3", "CH4", "CH5"):
        with st.expander(f"{channel} 通道（可选）", expanded=channel == "CH3"):
            uploaded = st.file_uploader(f"{channel} 文件", type=["csv", "txt"], key=f"{channel}-file")
            signal_column = st.text_input(f"{channel} 信号列", value="", key=f"{channel}-signal")
            time_column = st.text_input(f"{channel} 时间列（可选）", value="", key=f"{channel}-time")
            uploads[channel] = (uploaded, signal_column.strip() or None, time_column.strip() or None)
    left, right = st.columns(2)
    sampling_rate_hz = int(left.number_input("公共原始采样率 Hz", min_value=1, value=12000))
    rpm = float(right.number_input("公共转速 rpm", min_value=1.0, value=1500.0))
    if st.button("开始诊断", type="primary", use_container_width=True):
        channels = []
        for channel, (uploaded, signal_column, time_column) in uploads.items():
            if uploaded is None:
                continue
            channels.append(ChannelInput(channel, _persist(uploaded, channel), signal_column, time_column))
        try:
            request = build_multichannel_run_request(
                channels=tuple(channels), sampling_rate_hz=sampling_rate_hz, rpm=rpm
            )
            app_result = run_single_diagnosis(request)
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state["single_run_result"] = app_result
        st.session_state["latest_result_kind"] = "single"
    app_result = st.session_state.get("single_run_result")
    if app_result is not None:
        _render_result(st, app_result)


def batch_diagnosis_page() -> None:
    st = _st()
    st.markdown(build_page_header(title="V3批量诊断", subtitle="上传包含1至3个通道输入的 manifest.csv。"), unsafe_allow_html=True)
    uploaded = st.file_uploader("manifest.csv", type=["csv"])
    if st.button("开始批量诊断", type="primary"):
        if uploaded is None:
            st.error("请先上传 manifest.csv")
            return
        path = _persist(uploaded, "manifest")
        result = run_batch_diagnosis(AppBatchRunRequest(manifest_path=path))
        st.session_state["batch_run_result"] = result
    result = st.session_state.get("batch_run_result")
    if result is not None:
        st.metric("任务数", result.batch_result.total_count)
        st.metric("诊断成功数", result.batch_result.diagnosed_count)


def report_page() -> None:
    st = _st()
    st.markdown(build_page_header(title="诊断结果", subtitle="展示最近一次V3多通道诊断。"), unsafe_allow_html=True)
    result = st.session_state.get("single_run_result")
    if result is None:
        st.info("暂无单次诊断结果。")
        return
    _render_result(st, result)


def history_page() -> None:
    st = _st()
    st.markdown(build_page_header(title="V3历史记录", subtitle="仅读取独立的V3历史数据库。"), unsafe_allow_html=True)
    try:
        records = V3HistoryRepository(DEFAULT_HISTORY_DATABASE_PATH).list_all()
    except Exception as exc:
        st.error(f"历史记录无法读取：{exc}")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "时间": item.diagnosed_at,
                    "输入通道": "、".join(item.input_channels),
                    "诊断类别": display_label(item.predicted_label) if item.predicted_label else "-",
                    "采样率": item.sampling_rate_hz,
                    "转速": item.rpm,
                    "推理契约": item.contract_version,
                }
                for item in records
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def system_overview_page() -> None:
    st = _st()
    st.markdown(build_page_header(title="系统说明", subtitle="CatBoost43 V3 正式软件推理契约。"), unsafe_allow_html=True)
    st.markdown(
        """
        - CH3、CH4、CH5 使用三个独立 CatBoost 模型。
        - 信号处理：12000 Hz、5–5000 Hz零相位带通、4800点窗、2400点步长。
        - 特征：43维，db6三层小波包。
        - 窗口内和通道间均执行概率算术平均。
        - 部署模型仅用于软件推理，泛化评价使用新的独立数据。
        """
    )
