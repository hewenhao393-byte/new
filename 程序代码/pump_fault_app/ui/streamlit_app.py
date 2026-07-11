from __future__ import annotations

from typing import Any

from pump_fault_app.ui.branding import APP_SUBTITLE, APP_TITLE, build_research_style
from pump_fault_app.ui.pages import batch_diagnosis, report_view, single_diagnosis


def build_home_sections() -> dict[str, Any]:
    return {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "faults": (
            "正常",
            "转子不平衡",
            "联轴器不对中",
            "机械松动",
            "轴承故障",
            "汽蚀",
        ),
        "fault_cards": (
            {"title": "正常状态", "description": "稳定运行参考状态", "tone": "normal"},
            {"title": "转子不平衡", "description": "转频能量异常", "tone": "fault"},
            {"title": "联轴器不对中", "description": "倍频结构偏移", "tone": "fault"},
            {"title": "机械松动", "description": "连接刚度变化", "tone": "warning"},
            {"title": "轴承故障", "description": "冲击与调制特征", "tone": "fault"},
            {"title": "汽蚀", "description": "宽带水力扰动", "tone": "fault"},
        ),
        "pipeline": (
            "振动信号输入",
            "信号预处理",
            "特征提取",
            "BP神经网络",
            "六分类故障识别",
            "诊断报告输出",
        ),
        "parameters": {
            "默认采样率": "12000 Hz",
            "分析窗口": "2400点",
            "滑动步长": "1200点",
            "分析频带": "10–5000 Hz",
        },
    }


def build_navigation_items() -> list[dict[str, Any]]:
    return [
        {"title": "系统首页", "nav_label": "01 系统首页", "icon": None, "callable": render_home_page, "url_path": "home"},
        {"title": "单文件诊断", "nav_label": "02 单文件诊断", "icon": None, "callable": single_diagnosis.main, "url_path": "single"},
        {"title": "批量诊断", "nav_label": "03 批量诊断", "icon": None, "callable": batch_diagnosis.main, "url_path": "batch"},
        {"title": "诊断结果", "nav_label": "04 诊断结果", "icon": None, "callable": report_view.main, "url_path": "report"},
    ]


def build_global_style() -> str:
    return build_research_style()


def _st():
    import streamlit as st

    return st


def render_home_page(single_page: Any) -> None:
    st = _st()
    sections = build_home_sections()
    st.markdown(
        f'<div class="research-hero"><h1>{sections["title"]}</h1><div class="subtitle">{sections["subtitle"]}</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.6, 1.0), gap="large")
    with left:
        st.markdown("### 六类识别能力")
        st.markdown('<p class="section-intro">面向典型机械与水力状态的统一六分类诊断。</p>', unsafe_allow_html=True)
        for row in (sections["fault_cards"][:3], sections["fault_cards"][3:]):
            columns = st.columns(3)
            for column, card in zip(columns, row):
                with column:
                    st.markdown(
                        f'<div class="capability-card {card["tone"]}"><div class="title">{card["title"]}</div><div class="description">{card["description"]}</div></div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("### 智能诊断流程")
        flow_html = []
        for index, step in enumerate(sections["pipeline"]):
            if index:
                flow_html.append('<span class="flow-arrow">↓</span>')
            flow_html.append(f'<span class="flow-step">{step}</span>')
        st.markdown(
            f'<div class="flow-lane">{"".join(flow_html)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### 正式推理参数")
        st.markdown('<p class="section-intro">参数与 BP 六分类模型训练阶段保持一致。</p>', unsafe_allow_html=True)
        for label, value in sections["parameters"].items():
            st.markdown(
                f'<div class="app-card"><strong>{label}</strong><br><span class="section-intro">{value}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("进入单文件智能诊断", type="primary", use_container_width=True):
        st.switch_page(single_page)


def render_top_navigation(st: Any, page_by_path: dict[str, Any]) -> None:
    st.markdown('<div class="top-navigation-label">系统导航</div>', unsafe_allow_html=True)
    items = build_navigation_items()
    columns = st.columns(len(items), gap="small")
    for column, item in zip(columns, items):
        with column:
            if st.button(
                item["nav_label"],
                key=f'top-nav-{item["url_path"]}',
                use_container_width=True,
            ):
                st.switch_page(page_by_path[item["url_path"]])


def main() -> None:
    st = _st()
    st.set_page_config(page_title=APP_TITLE, page_icon="■", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(build_global_style(), unsafe_allow_html=True)
    items = build_navigation_items()
    item_by_path = {item["url_path"]: item for item in items}
    single_item = item_by_path["single"]
    single_page = st.Page(
        single_item["callable"],
        title=single_item["title"],
        icon=single_item["icon"],
        url_path=single_item["url_path"],
    )
    page_by_path = {
        "home": st.Page(
            lambda: render_home_page(single_page),
            title=item_by_path["home"]["title"],
            icon=item_by_path["home"]["icon"],
            url_path=item_by_path["home"]["url_path"],
            default=True,
        ),
        "single": single_page,
        **{
            item["url_path"]: st.Page(
                item["callable"],
                title=item["title"],
                icon=item["icon"],
                url_path=item["url_path"],
            )
            for item in items
            if item["url_path"] not in {"home", "single"}
        },
    }
    pages = [page_by_path[item["url_path"]] for item in items]
    navigation = st.navigation(pages, position="sidebar")
    render_top_navigation(st, page_by_path)
    navigation.run()


if __name__ == "__main__":
    main()
