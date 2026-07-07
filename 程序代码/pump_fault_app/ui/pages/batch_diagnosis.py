from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

from pump_fault_app.services import AppBatchRunRequest, run_batch_diagnosis


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
    return AppBatchRunRequest(
        manifest_path=manifest_path,
        export_root=export_root,
    )


def build_batch_table_rows(summaries: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        confidence = summary.get("confidence")
        rows.append(
            {
                "文件": str(summary.get("file_name") or "-"),
                "预测类别": str(summary.get("diagnosis_label") or "-"),
                "置信度": "-" if confidence is None else f"{float(confidence):.3f}",
                "状态": str(summary.get("status") or "-"),
            }
        )
    return rows


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
    st.download_button(
        "下载批量 JSON",
        data=export_result.json_path.read_bytes(),
        file_name=export_result.json_path.name,
        mime="application/json",
    )
    st.download_button(
        "下载批量 CSV",
        data=export_result.csv_path.read_bytes(),
        file_name=export_result.csv_path.name,
        mime="text/csv",
    )


def main() -> None:
    st = _st()
    st.title("批量诊断")
    mode = st.radio("输入方式", ("多个振动文件", "manifest.csv"), horizontal=True)

    result = None
    if mode == "多个振动文件":
        uploaded_files = st.file_uploader("上传多个振动文件", type=["csv", "txt", "wav"], accept_multiple_files=True)
        sampling_rate_hz = int(st.number_input("采样率 Hz", min_value=1, value=12000, step=100))
        rpm = float(st.number_input("转速 rpm", min_value=1.0, value=1450.0, step=10.0))
        signal_column = st.text_input("信号列名（可选）", value="通道4")
        time_column = st.text_input("时间列名（可选）", value="time")
        if st.button("开始批量诊断", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("请先上传文件。")
                return
            saved = _persist_uploaded_files(uploaded_files, prefix="pump_batch_files_")
            result = run_batch_diagnosis(
                build_multi_file_batch_request(
                    file_paths=saved,
                    sampling_rate_hz=sampling_rate_hz,
                    rpm=rpm,
                    signal_column=signal_column,
                    time_column=time_column,
                    export_root=Path(tempfile.mkdtemp(prefix="pump_batch_export_")),
                )
            )
            st.session_state["batch_run_result"] = result
    else:
        manifest_file = st.file_uploader("上传 manifest.csv", type=["csv"])
        if st.button("开始批量诊断", type="primary", use_container_width=True):
            if manifest_file is None:
                st.error("请先上传 manifest.csv。")
                return
            manifest_path = _persist_manifest(manifest_file, prefix="pump_manifest_")
            result = run_batch_diagnosis(
                build_batch_manifest_request(
                    manifest_path=manifest_path,
                    export_root=Path(tempfile.mkdtemp(prefix="pump_batch_export_")),
                )
            )
            st.session_state["batch_run_result"] = result

    if result is None:
        return

    st.subheader("批量结果")
    summaries = [summary.as_dict() for summary in result.batch_result.summaries]
    st.dataframe(build_batch_table_rows(summaries), use_container_width=True)
    st.json(
        {
            "total_count": result.batch_result.total_count,
            "diagnosed_count": result.batch_result.diagnosed_count,
            "rejected_count": result.batch_result.rejected_count,
            "input_error_count": result.batch_result.input_error_count,
        }
    )
    _render_downloads(st, result.export_result)


if __name__ == "__main__":
    main()
