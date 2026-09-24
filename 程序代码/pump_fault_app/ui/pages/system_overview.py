from __future__ import annotations

from html import escape
from typing import Any

from pump_fault_app.presentation.system_overview import build_system_overview
from pump_fault_app.ui.branding import build_page_header


def _st():
    import streamlit as st

    return st


def main() -> None:
    st = _st()
    overview = build_system_overview()
    st.markdown(
        """
        <style>
        .principle-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            padding: 1rem;
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 10px;
        }
        .principle-flow-step {
            padding: 0.65rem 0.8rem;
            color: #1d4e6d;
            background: #f2f8fc;
            border-left: 3px solid #4b9bc4;
            border-radius: 5px;
            font-weight: 600;
            white-space: nowrap;
        }
        .principle-flow-arrow { color: #6f92aa; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        build_page_header(title=overview["title"], subtitle=overview["subtitle"]),
        unsafe_allow_html=True,
    )

    st.markdown("### 诊断流程")
    flow_html = []
    for index, step in enumerate(overview["pipeline"]):
        if index:
            flow_html.append('<span class="principle-flow-arrow">→</span>')
        flow_html.append(f'<span class="principle-flow-step">{escape(step)}</span>')
    st.markdown(f'<div class="principle-flow">{"".join(flow_html)}</div>', unsafe_allow_html=True)

    st.markdown("### 21维振动特征的工程含义")
    feature_groups = overview["feature_groups"]
    for group_row in (feature_groups[:2], feature_groups[2:]):
        columns = st.columns(2, gap="large")
        for column, group in zip(columns, group_row):
            with column:
                st.markdown(
                    '<div class="app-card" style="padding:1rem 1.05rem;min-height:13rem;margin-bottom:0.8rem">'
                    f'<strong style="color:#1d4e6d">{escape(group["title"])}</strong>'
                    f'<div class="section-intro" style="margin-top:0.25rem">{len(group["features"])}项特征</div>'
                    f'<p style="color:#486581;line-height:1.65">{escape(group["explanation"])}</p>'
                    f'<div class="section-intro">{escape("、".join(group["features"]))}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("### BP模型与诊断结果融合")
    model_columns = st.columns(2, gap="large")
    for column, key in zip(model_columns, ("窗口级预测", "多窗口概率融合")):
        with column:
            st.markdown(
                '<div class="app-card" style="padding:1rem 1.05rem;min-height:8rem">'
                f'<strong style="color:#1d4e6d">{escape(key)}</strong>'
                f'<p style="color:#486581;line-height:1.65">{escape(overview["model"][key])}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div class="input-note" style="margin-top:0.8rem"><strong>{escape(overview["model"]["诊断方式"])}</strong>'
        f'：最终输出类别为{escape("、".join(overview["labels"]))}。</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 典型故障—振动表现—敏感特征对应关系")
    st.dataframe(
        overview["fault_feature_rows"],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### 正式推理参数")
    st.markdown(
        '<p class="section-intro">以下参数直接读取正式推理契约，仅调整展示宽度，不改变模型输入与处理流程。</p>',
        unsafe_allow_html=True,
    )
    for parameter_row in overview["parameter_rows"]:
        columns = st.columns(len(parameter_row), gap="large")
        for column, item in zip(columns, parameter_row):
            with column:
                st.markdown(
                    '<div class="app-card" style="padding:0.9rem 1rem;margin-bottom:0.8rem;min-height:5.4rem">'
                    f'<strong style="color:#1d4e6d">{escape(item["label"])}</strong><br>'
                    f'<span style="color:#486581;font-size:1.05rem;white-space:normal">{escape(item["value"])}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
