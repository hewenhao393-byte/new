from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pump_fault_app.services import AppBatchRunRequest, run_batch_diagnosis
from pump_fault_app.presentation.batch_diagnosis import (
    build_batch_filter_options,
    build_batch_table_rows,
    build_batch_task_statistics,
)
from pump_fault_app.ui.branding import build_page_header


def build_multi_file_batch_request(
    *,
    file_paths: tuple[Path, ...],
    sampling_rate_hz: int,
    rpm: float,
    signal_column: str | None = None,
    time_column: str | None = None,
    export_root: Path | None = None,
) -> AppBatchRunRequest:
    return AppBatchRunRequest(
        file_paths=file_paths,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        signal_column=signal_column or None,
        time_column=time_column or None,
        export_root=export_root,
    )


def build_batch_manifest_request(*, manifest_path: Path, export_root: Path | None = None) -> AppBatchRunRequest:
    return AppBatchRunRequest(manifest_path=manifest_path, export_root=export_root)


def _st():
    import streamlit as st

    return st


def _persist_uploaded_files(uploaded_files: Iterable[Any], *, prefix: str) -> tuple[Path, ...]:
    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    saved: list[Path] = []
    for uploaded_file in uploaded_files:
        target = workdir / uploaded_file.name
        target.write_bytes(uploaded_file.getbuffer())
        saved.append(target)
    return tuple(saved)


def _persist_manifest(uploaded_file: Any, *, prefix: str) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    target = workdir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_downloads(st: Any, export_result: Any) -> None:
    if export_result is None:
        return
    left, right = st.columns(2)
    with left:
        st.download_button(
            "下载批量 JSON",
            data=export_result.json_path.read_bytes(),
            file_name=export_result.json_path.name,
            mime="application/json",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "下载批量 CSV",
            data=export_result.csv_path.read_bytes(),
            file_name=export_result.csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st = _st()
    st.markdown(
        build_page_header(
            title="批量诊断任务",
            subtitle="支持多文件上传或 manifest.csv 清单，统一调用正式批量诊断服务。",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("#### 批量任务输入")

    mode = st.radio("输入方式", ("多个振动文件", "manifest.csv"), horizontal=True)
    result = None

    if mode == "多个振动文件":
        uploaded_files = st.file_uploader("上传多个振动文件", type=["csv", "txt", "wav"], accept_multiple_files=True)
        col1, col2 = st.columns(2)
        with col1:
            sampling_rate_hz = int(st.number_input("采样率 Hz", min_value=1, value=12000, step=100))
        with col2:
            rpm = float(st.number_input("转速 rpm", min_value=1.0, value=1450.0, step=10.0))
        with st.expander("高级设置", expanded=False):
            signal_column = st.text_input("振动信号列（选填，留空时自动识别）", value="")
            time_column = st.text_input("时间列（选填，留空时自动识别）", value="")
            export_enabled = st.checkbox("导出设置", value=True)
        if st.button("开始批量诊断", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("请先上传振动文件。")
                return
            saved = _persist_uploaded_files(uploaded_files, prefix="pump_batch_files_")
            result = run_batch_diagnosis(
                build_multi_file_batch_request(
                    file_paths=saved,
                    sampling_rate_hz=sampling_rate_hz,
                    rpm=rpm,
                    signal_column=signal_column,
                    time_column=time_column,
                    export_root=Path(tempfile.mkdtemp(prefix="pump_batch_export_")) if export_enabled else None,
                )
            )
            st.session_state["batch_run_result"] = result
            st.session_state["latest_result_kind"] = "batch"
    else:
        manifest_file = st.file_uploader("上传 manifest.csv", type=["csv"])
        with st.expander("高级设置", expanded=False):
            export_enabled = st.checkbox("导出设置", value=True)
        if st.button("开始批量诊断", type="primary", use_container_width=True):
            if manifest_file is None:
                st.error("请先上传 manifest.csv。")
                return
            manifest_path = _persist_manifest(manifest_file, prefix="pump_manifest_")
            result = run_batch_diagnosis(
                build_batch_manifest_request(
                    manifest_path=manifest_path,
                    export_root=Path(tempfile.mkdtemp(prefix="pump_batch_export_")) if export_enabled else None,
                )
            )
            st.session_state["batch_run_result"] = result
            st.session_state["latest_result_kind"] = "batch"

    if result is None:
        return

    summaries = [summary.as_dict() for summary in result.batch_result.summaries]
    statistics = build_batch_task_statistics(summaries)
    st.subheader("批量任务统计总览")
    st.markdown('<p class="task-stat-caption">基于当前批量任务的已有诊断结果汇总。</p>', unsafe_allow_html=True)
    statistic_columns = st.columns(4)
    for column, (label, value) in zip(statistic_columns, statistics.items()):
        with column:
            st.metric(label, value)

    filter_options = build_batch_filter_options(summaries)
    selected_label = st.selectbox("按预测类别筛选", filter_options, index=0)
    if selected_label != "全部":
        summaries = [summary for summary in summaries if summary.get("diagnosis_label") == selected_label]

    st.subheader("批量诊断结果明细")
    st.dataframe(pd.DataFrame(build_batch_table_rows(summaries)), use_container_width=True, hide_index=True)
    _render_downloads(st, result.export_result)


if __name__ == "__main__":
    main()
