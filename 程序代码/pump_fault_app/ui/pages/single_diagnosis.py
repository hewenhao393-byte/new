from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis

_DISPLAY_LABELS = (
    ("正常", "正常"),
    ("转子不平衡", "转子不平衡"),
    ("联轴器不对中", "联轴器不对中"),
    ("松动", "机械松动"),
    ("轴承故障", "轴承故障"),
    ("汽蚀", "汽蚀"),
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


def build_probability_rows(top_probabilities: tuple[dict[str, float | str], ...]) -> list[dict[str, float | str]]:
    probability_map = {
        str(item["label"]): float(item["probability"])
        for item in top_probabilities
    }
    return [
        {
            "label": display_label,
            "probability": round(probability_map.get(internal_label, 0.0), 6),
        }
        for internal_label, display_label in _DISPLAY_LABELS
    ]


def build_single_summary_items(
    *,
    quality_text: str,
    window_count: int | None,
    elapsed_seconds: float,
    message: str,
    warning_count: int,
) -> dict[str, str]:
    return {
        "信号质量": quality_text,
        "窗口数量": "-" if window_count is None else str(window_count),
        "推理耗时": f"{elapsed_seconds:.3f} s",
        "关键输出": message,
        "运行告警数": str(warning_count),
    }


def _st():
    import streamlit as st

    return st


def _persist_uploaded_file(uploaded_file: Any, *, prefix: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    target = workdir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    if suffix.lower() == ".wav":
        pass
    return target


def _render_downloads(st: Any, export_result: Any) -> None:
    if export_result is None:
        return
    st.download_button(
        "下载 JSON 结果",
        data=export_result.json_path.read_bytes(),
        file_name=export_result.json_path.name,
        mime="application/json",
    )
    st.download_button(
        "下载 CSV 结果",
        data=export_result.csv_path.read_bytes(),
        file_name=export_result.csv_path.name,
        mime="text/csv",
    )


def main() -> None:
    st = _st()
    st.title("单文件诊断")
    st.caption("上传单条振动记录并调用统一 service API 完成正式六分类诊断。")

    uploaded_file = st.file_uploader("振动文件", type=["csv", "txt", "wav"])
    sampling_rate_hz = int(st.number_input("采样率 Hz", min_value=1, value=12000, step=100))
    rpm = float(st.number_input("转速 rpm", min_value=1.0, value=1450.0, step=10.0))
    signal_column = st.text_input("信号列名（可选）", value="通道4")
    time_column = st.text_input("时间列名（可选）", value="time")
    device_id = st.text_input("设备编号（可选）", value="")
    measurement_position = st.text_input("测点位置（可选）", value="")

    if uploaded_file is not None and Path(uploaded_file.name).suffix.lower() == ".wav":
        st.warning("当前正式后端已预留 wav 上传入口；若运行时提示格式不支持，请改用 csv/txt 或后续补充 wav 读取能力。")

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
                export_root=Path(tempfile.mkdtemp(prefix="pump_single_export_")),
            )
        )
        st.session_state["single_run_result"] = result
        elapsed = time.perf_counter() - started
        summary = result.summary

        if summary.success:
            st.success("诊断完成")
        else:
            st.error(summary.message)

        st.subheader("诊断结果")
        st.write(f"最终故障类别：{summary.diagnosis_label or '-'}")
        st.write(f"预测置信度：{'-' if summary.confidence is None else f'{summary.confidence:.3f}'}")

        st.subheader("六分类概率分布")
        st.dataframe(build_probability_rows(summary.top_probabilities), use_container_width=True)

        quality_text = (
            result.inference_result.quality_report.quality_level
            if result.inference_result.quality_report is not None
            else "-"
        )
        st.subheader("诊断摘要")
        st.json(
            build_single_summary_items(
                quality_text=quality_text,
                window_count=summary.window_count,
                elapsed_seconds=elapsed,
                message=summary.message,
                warning_count=len(summary.runtime_alerts),
            )
        )
        if summary.runtime_alerts:
            st.subheader("运行告警")
            st.dataframe(summary.runtime_alerts, use_container_width=True)

        _render_downloads(st, result.export_result)


if __name__ == "__main__":
    main()
