from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis
from pump_fault_app.presentation.single_diagnosis import (
    build_probability_rows,
    build_upload_signal_info,
    build_single_summary_items,
    build_single_visual_availability,
    build_spectrum_rows,
    build_structured_summary_rows,
    build_time_domain_rows,
    build_wavelet_packet_rows,
    build_window_distribution_rows,
)

def build_single_run_request(
    *,
    file_path: Path,
    sampling_rate_hz: int,
    rpm: float,
    signal_column: str | None = None,
    time_column: str | None = None,
    device_id: str | None = None,
    measurement_position: str | None = None,
    model_bundle_path: Path | None = None,
    export_root: Path | None = None,
) -> AppSingleRunRequest:
    return AppSingleRunRequest(
        file_path=file_path,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        signal_column=signal_column or None,
        time_column=time_column or None,
        device_id=device_id or None,
        measurement_position=measurement_position or None,
        model_bundle_path=model_bundle_path,
        export_root=export_root,
    )


def get_single_advanced_field_labels() -> dict[str, str]:
    return {
        "signal_column": "振动信号列（选填，留空时自动识别）",
        "time_column": "时间列（选填，留空时自动识别）",
        "device_id": "设备编号（选填）",
        "measurement_position": "测点位置（选填，例如：泵驱动端水平）",
        "export": "导出设置",
    }


def _st():
    import streamlit as st

    return st


def _alt():
    import altair as alt

    return alt


def _persist_uploaded_file(uploaded_file: Any, *, prefix: str) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    target = workdir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_probability_chart(st: Any, rows: list[dict[str, float | str]]) -> None:
    alt = _alt()
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#0f5b8d")
        .encode(
            x=alt.X("probability:Q", title="概率"),
            y=alt.Y("label:N", sort=[row["label"] for row in rows], title="类别"),
            tooltip=["label", "probability"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_window_distribution(st: Any, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        return
    alt = _alt()
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#157a6e")
        .encode(
            x=alt.X("label:N", title="类别"),
            y=alt.Y("count:Q", title="窗口数"),
            tooltip=["label", "count"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_line_chart(
    st: Any,
    frame: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    color: str,
    height: int = 280,
) -> None:
    alt = _alt()
    chart = (
        alt.Chart(frame)
        .mark_line(color=color)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[x_field, y_field],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_wavelet_packet(st: Any, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    alt = _alt()
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#2f6f4f")
        .encode(
            x=alt.X("band_label:N", title="频带"),
            y=alt.Y("energy_ratio:Q", title="能量占比"),
            tooltip=["band_label", "energy_ratio"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_downloads(st: Any, export_result: Any) -> None:
    if export_result is None:
        return
    left, right = st.columns(2)
    with left:
        st.download_button(
            "下载 JSON 报告",
            data=export_result.json_path.read_bytes(),
            file_name=export_result.json_path.name,
            mime="application/json",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "下载 CSV 报告",
            data=export_result.csv_path.read_bytes(),
            file_name=export_result.csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st = _st()
    labels = get_single_advanced_field_labels()

    st.title("单文件诊断")
    st.caption("上传一条振动记录并调用统一诊断服务完成正式六分类判断。")

    uploaded_file = st.file_uploader("振动数据文件", type=["csv", "txt", "wav"])
    st.markdown(
        '<div class="input-note">输入信号采样率可不同于模型标准采样率，系统会在正式推理中自动完成重采样处理。</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        sampling_rate_hz = int(st.number_input("采样率 Hz", min_value=1, value=12000, step=100))
    with col2:
        rpm = float(st.number_input("转速 rpm", min_value=1.0, value=1450.0, step=10.0))

    with st.expander("高级设置", expanded=False):
        signal_column = st.text_input(labels["signal_column"], value="")
        time_column = st.text_input(labels["time_column"], value="")
        device_id = st.text_input(labels["device_id"], value="")
        measurement_position = st.text_input(labels["measurement_position"], value="")
        export_enabled = st.checkbox(labels["export"], value=True, help="勾选后生成可下载的 JSON/CSV 结果文件。")

    if uploaded_file is not None:
        st.subheader("信号文件信息")
        st.dataframe(
            build_upload_signal_info(
                file_name=uploaded_file.name,
                file_size_bytes=getattr(uploaded_file, "size", None),
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                signal_column=signal_column,
                time_column=time_column,
                measurement_position=measurement_position,
            ),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("开始诊断", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("请先上传振动文件。")
            return

        persisted = _persist_uploaded_file(uploaded_file, prefix="pump_single_")
        started = time.perf_counter()
        result = run_single_diagnosis(
            build_single_run_request(
                file_path=persisted,
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                signal_column=signal_column,
                time_column=time_column,
                device_id=device_id,
                measurement_position=measurement_position,
                export_root=Path(tempfile.mkdtemp(prefix="pump_single_export_")) if export_enabled else None,
            )
        )
        st.session_state["single_run_result"] = result
        st.session_state["latest_result_kind"] = "single"
        elapsed = time.perf_counter() - started
        summary = result.summary
        visual_availability = build_single_visual_availability(result)
        diagnosis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if summary.success:
            st.success("诊断完成")
        else:
            st.error(summary.message)

        quality_text = result.inference_result.quality_report.quality_level if result.inference_result.quality_report is not None else "-"
        metric_cols = st.columns(4)
        metric_cols[0].metric("预测故障类别", summary.diagnosis_label or "-")
        metric_cols[1].metric("综合置信度", "-" if summary.confidence is None else f"{summary.confidence:.3f}")
        metric_cols[2].metric("有效窗口数", "-" if summary.window_count is None else str(summary.window_count))
        metric_cols[3].metric("信号质量", quality_text)

        if visual_availability["probability_chart"]:
            st.subheader("六分类概率分布")
            _render_probability_chart(st, build_probability_rows(summary.top_probabilities))

        if visual_availability["window_distribution"]:
            rows = build_window_distribution_rows(result.inference_result.window_predictions)
            if rows:
                st.subheader("窗口级预测类别分布")
                _render_window_distribution(st, rows)

        if visual_availability["waveform"] and result.visualization is not None and result.visualization.time_domain is not None:
            st.subheader("时域波形（展示数据）")
            _render_line_chart(
                st,
                build_time_domain_rows(result.visualization.time_domain),
                x_field="time_seconds",
                y_field="amplitude",
                x_title="时间 / s",
                y_title="幅值",
                color="#0f5b8d",
            )

        if visual_availability["spectrum"] and result.visualization is not None and result.visualization.frequency_spectrum is not None:
            st.subheader("0–5000 Hz 频谱图（展示数据）")
            _render_line_chart(
                st,
                build_spectrum_rows(result.visualization.frequency_spectrum),
                x_field="frequency_hz",
                y_field="amplitude",
                x_title="频率 / Hz",
                y_title="幅值",
                color="#157a6e",
            )
        with st.expander("详细分析", expanded=False):
            if visual_availability["envelope"] and result.visualization is not None and result.visualization.envelope_spectrum is not None:
                st.markdown("#### 包络谱（展示数据）")
                _render_line_chart(
                    st,
                    build_spectrum_rows(result.visualization.envelope_spectrum),
                    x_field="frequency_hz",
                    y_field="amplitude",
                    x_title="频率 / Hz",
                    y_title="幅值",
                    color="#a05a2c",
                )
            if visual_availability["wavelet"] and result.visualization is not None and result.visualization.wavelet_packet_energy is not None:
                st.markdown("#### 小波包能量占比（展示数据）")
                _render_wavelet_packet(st, build_wavelet_packet_rows(result.visualization.wavelet_packet_energy))

        st.subheader("诊断摘要")
        st.dataframe(
            build_structured_summary_rows(summary, result.inference_result, diagnosis_time),
            use_container_width=True,
            hide_index=True,
        )
        summary_items = build_single_summary_items(
            quality_text=quality_text,
            window_count=summary.window_count,
            elapsed_seconds=elapsed,
            message=summary.message,
            warning_count=len(summary.runtime_alerts),
        )
        st.caption("关键输出信息")
        st.dataframe(
            [{"项目": key, "内容": value} for key, value in summary_items.items()],
            use_container_width=True,
            hide_index=True,
        )

        if summary.runtime_alerts:
            st.info("信号处理状态：已完成数值稳定性保护处理，不影响诊断结果。")
            with st.expander(f"查看详细运行告警（{len(summary.runtime_alerts)} 条）", expanded=False):
                st.dataframe(summary.runtime_alerts, use_container_width=True, hide_index=True)

        if result.visualization is not None and result.visualization.warnings:
            st.warning("部分可视化数据生成失败，已保留正式诊断结果。")
            st.dataframe(
                [{"告警": warning} for warning in result.visualization.warnings],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("报告导出")
        _render_downloads(st, result.export_result)


if __name__ == "__main__":
    main()
