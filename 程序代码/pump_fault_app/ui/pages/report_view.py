from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from pump_fault_app.reporting import build_single_report_view_data
from pump_fault_app.services import AppBatchRunResult, AppSingleRunResult, export_single_diagnosis_report
from pump_fault_app.presentation.single_diagnosis import (
    build_spectrum_rows,
    build_structured_summary_rows,
    build_time_domain_rows,
    build_wavelet_packet_rows,
)
from pump_fault_app.presentation.batch_diagnosis import build_batch_table_rows
from pump_fault_app.presentation.report_view import build_diagnosis_highlight
from pump_fault_app.ui.streamlit_app import APP_SUBTITLE, APP_TITLE


def build_report_empty_message() -> str:
    return "暂无诊断结果，请先完成单文件诊断或批量诊断。"


def build_report_overview(
    single_result: AppSingleRunResult | None,
    batch_result: AppBatchRunResult | None,
    latest_kind: str | None = None,
) -> dict[str, Any]:
    active_kind = None
    if latest_kind == "batch" and batch_result is not None:
        active_kind = "batch"
    elif latest_kind == "single" and single_result is not None:
        active_kind = "single"
    elif single_result is not None:
        active_kind = "single"
    elif batch_result is not None:
        active_kind = "batch"
    return {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "active_kind": active_kind,
    }


def _st():
    import streamlit as st

    return st


def _alt():
    import altair as alt

    return alt


def _render_line_chart(st: Any, frame: pd.DataFrame, *, x_field: str, y_field: str, x_title: str, y_title: str, color: str) -> None:
    alt = _alt()
    chart = (
        alt.Chart(frame)
        .mark_line(color=color)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[x_field, y_field],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_bar_chart(st: Any, frame: pd.DataFrame, *, x_field: str, y_field: str, x_title: str, y_title: str, color: str) -> None:
    alt = _alt()
    chart = (
        alt.Chart(frame)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x_field}:N", title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[x_field, y_field],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_diagnosis_highlight(st: Any, *, label: str | None, confidence: str, grade: str | None) -> None:
    highlight = build_diagnosis_highlight(label, confidence, grade)
    st.markdown(
        f'<div class="diagnosis-highlight {highlight["tone"]}">'
        '<div class="label">最终诊断</div>'
        f'<div class="result">{escape(highlight["label"])}</div>'
        f'<div class="meta">置信度：{escape(highlight["confidence"])} &nbsp;&nbsp; 诊断状态：{escape(highlight["status"])} &nbsp;&nbsp; 诊断等级：{escape(highlight["grade"])}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_single_report(st: Any, result: AppSingleRunResult) -> None:
    summary = result.summary
    view_data = build_single_report_view_data(result)
    _render_diagnosis_highlight(
        st,
        label=view_data.conclusion.diagnosis_label,
        confidence=view_data.conclusion.top_probability,
        grade=view_data.conclusion.diagnosis_grade,
    )
    visual_rows = {
        "time_domain": None if view_data.time_domain is None else build_time_domain_rows(view_data.time_domain),
        "frequency_spectrum": None if view_data.frequency_spectrum is None else build_spectrum_rows(view_data.frequency_spectrum),
        "envelope_spectrum": None if view_data.envelope_spectrum is None else build_spectrum_rows(view_data.envelope_spectrum),
        "wavelet_packet_energy": None if view_data.wavelet_packet_energy is None else build_wavelet_packet_rows(view_data.wavelet_packet_energy),
    }
    st.subheader("诊断基本信息")
    st.dataframe(
        pd.DataFrame([{"项目": item.label, "内容": item.value} for item in view_data.basic_info]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("诊断结论")
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "最终类别", "内容": view_data.conclusion.diagnosis_label},
                {"项目": "最高平均概率", "内容": view_data.conclusion.top_probability},
                {"项目": "第二可能类别", "内容": view_data.conclusion.second_label},
                {"项目": "概率差", "内容": view_data.conclusion.probability_gap},
                {"项目": "窗口一致率", "内容": view_data.conclusion.window_consistency},
                {"项目": "诊断等级", "内容": view_data.conclusion.diagnosis_grade},
                {"项目": "诊断建议", "内容": view_data.conclusion.diagnosis_advice},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if view_data.probabilities:
        st.subheader("六分类概率")
        probability_frame = pd.DataFrame(
            [{"label": item.label, "probability": item.probability} for item in view_data.probabilities]
        )
        _render_bar_chart(st, probability_frame, x_field="label", y_field="probability", x_title="类别", y_title="概率", color="#0f5b8d")

    if view_data.window_distribution:
        st.subheader("窗口级预测分布")
        _render_bar_chart(
            st,
            pd.DataFrame([{"label": item.label, "count": item.window_count} for item in view_data.window_distribution]),
            x_field="label",
            y_field="count",
            x_title="类别",
            y_title="窗口数",
            color="#157a6e",
        )

    st.subheader("信号质量")
    st.write(view_data.quality_text)

    st.subheader("信号分析图")
    if visual_rows["time_domain"] is not None:
        st.markdown("#### 时域波形（展示数据）")
        _render_line_chart(st, visual_rows["time_domain"], x_field="time_seconds", y_field="amplitude", x_title="时间 / s", y_title="幅值", color="#0f5b8d")
    if visual_rows["frequency_spectrum"] is not None:
        st.markdown("#### 0–5000 Hz 频谱（展示数据）")
        _render_line_chart(st, visual_rows["frequency_spectrum"], x_field="frequency_hz", y_field="amplitude", x_title="频率 / Hz", y_title="幅值", color="#157a6e")
    if visual_rows["envelope_spectrum"] is not None:
        st.markdown("#### 包络谱（展示数据）")
        _render_line_chart(st, visual_rows["envelope_spectrum"], x_field="frequency_hz", y_field="amplitude", x_title="频率 / Hz", y_title="幅值", color="#a05a2c")
    if visual_rows["wavelet_packet_energy"] is not None:
        st.markdown("#### 小波包能量占比（展示数据）")
        _render_bar_chart(st, pd.DataFrame(visual_rows["wavelet_packet_energy"]), x_field="band_label", y_field="energy_ratio", x_title="频带", y_title="能量占比", color="#2f6f4f")

    if view_data.visualization_availability.messages:
        st.subheader("可视化告警")
        st.dataframe(
            pd.DataFrame([{"告警": item} for item in view_data.visualization_availability.messages]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("最终结果")
    st.metric("预测类别", summary.diagnosis_label or "-")
    st.metric("综合置信度", "-" if summary.confidence is None else f"{summary.confidence:.3f}")

    st.subheader("诊断摘要")
    st.dataframe(
        pd.DataFrame([{"项目": item.label, "内容": item.value} for item in view_data.summary_items]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("处理参数")
    st.dataframe(pd.DataFrame(build_structured_summary_rows(summary, result.inference_result, "-")), use_container_width=True, hide_index=True)

    if result.export_result is not None:
        st.subheader("导出")
        left, middle, right = st.columns(3)
        with left:
            st.download_button("下载 JSON 报告", data=result.export_result.json_path.read_bytes(), file_name=result.export_result.json_path.name, mime="application/json", use_container_width=True)
        with middle:
            st.download_button("下载 CSV 报告", data=result.export_result.csv_path.read_bytes(), file_name=result.export_result.csv_path.name, mime="text/csv", use_container_width=True)
        with right:
            try:
                docx_dir = Path(tempfile.mkdtemp(prefix="pump_single_docx_"))
                docx_path = export_single_diagnosis_report(result, docx_dir / "single_diagnosis_report.docx", format="docx")
                st.download_button(
                    "导出 Word 报告",
                    data=docx_path.read_bytes(),
                    file_name=docx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Word 报告导出失败：{exc}")
    else:
        st.subheader("导出")
        try:
            docx_dir = Path(tempfile.mkdtemp(prefix="pump_single_docx_"))
            docx_path = export_single_diagnosis_report(result, docx_dir / "single_diagnosis_report.docx", format="docx")
            st.download_button(
                "导出 Word 报告",
                data=docx_path.read_bytes(),
                file_name=docx_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Word 报告导出失败：{exc}")


def _render_batch_report(st: Any, result: AppBatchRunResult) -> None:
    summaries = [summary.as_dict() for summary in result.batch_result.summaries]
    st.subheader("诊断基本信息")
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "文件总数", "内容": result.batch_result.total_count},
                {"项目": "成功诊断数", "内容": result.batch_result.diagnosed_count},
                {"项目": "质量拒绝数", "内容": result.batch_result.rejected_count},
                {"项目": "输入错误数", "内容": result.batch_result.input_error_count},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("最终结果")
    st.dataframe(
        pd.DataFrame(build_batch_table_rows(summaries)),
        use_container_width=True,
        hide_index=True,
    )
    file_options = [summary.get("file_name") or f"记录{index + 1}" for index, summary in enumerate(summaries)]
    selected_name = st.selectbox("查看单条记录详情", file_options, index=0)
    selected_summary = summaries[file_options.index(selected_name)]

    st.subheader("信号质量")
    st.write(selected_summary.get("signal_quality") or _infer_signal_quality_from_status(str(selected_summary.get("status") or "")))

    if selected_summary.get("top_probabilities"):
        st.subheader("六分类概率")
        st.dataframe(
            pd.DataFrame(build_probability_rows(tuple(selected_summary["top_probabilities"]))),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("处理参数")
    parameter_rows = []
    if selected_summary.get("sampling_rate_hz") is not None:
        parameter_rows.append({"项目": "采样率", "内容": f'{selected_summary["sampling_rate_hz"]} Hz'})
    if selected_summary.get("rpm") is not None:
        parameter_rows.append({"项目": "转速", "内容": f'{float(selected_summary["rpm"]):.1f} rpm'})
    if parameter_rows:
        st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True, hide_index=True)

    st.subheader("诊断摘要")
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "文件名称", "内容": selected_summary.get("file_name") or "-"},
                {"项目": "设备编号", "内容": selected_summary.get("device_id") or "-"},
                {"项目": "预测类别", "内容": selected_summary.get("diagnosis_label") or "-"},
                {"项目": "综合置信度", "内容": "-" if selected_summary.get("confidence") is None else f'{float(selected_summary["confidence"]):.3f}'},
                {"项目": "状态", "内容": selected_summary.get("status") or "-"},
                {"项目": "摘要", "内容": selected_summary.get("message") or "-"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if result.export_result is not None:
        st.subheader("导出")
        left, right = st.columns(2)
        with left:
            st.download_button("下载批量 JSON", data=result.export_result.json_path.read_bytes(), file_name=result.export_result.json_path.name, mime="application/json", use_container_width=True)
        with right:
            st.download_button("下载批量 CSV", data=result.export_result.csv_path.read_bytes(), file_name=result.export_result.csv_path.name, mime="text/csv", use_container_width=True)


def _infer_signal_quality_from_status(status: str) -> str:
    if status == "diagnosed":
        return "pass"
    if status == "rejected":
        return "rejected"
    if status == "input_error":
        return "unavailable"
    return "-"


def main() -> None:
    st = _st()
    overview = build_report_overview(
        st.session_state.get("single_run_result"),
        st.session_state.get("batch_run_result"),
        st.session_state.get("latest_result_kind"),
    )
    st.title("诊断结果")
    st.caption(overview["subtitle"])

    if overview["active_kind"] is None:
        st.info(build_report_empty_message())
        return

    if overview["active_kind"] == "single":
        _render_single_report(st, st.session_state["single_run_result"])
    else:
        _render_batch_report(st, st.session_state["batch_run_result"])


if __name__ == "__main__":
    main()
