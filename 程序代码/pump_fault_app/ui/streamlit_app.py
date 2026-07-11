from __future__ import annotations

from typing import Any

from pump_fault_app.ui.branding import APP_SUBTITLE, APP_TITLE
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
        {"title": "系统首页", "icon": "🏠", "callable": render_home_page, "url_path": "home"},
        {"title": "单文件诊断", "icon": "📈", "callable": single_diagnosis.main, "url_path": "single"},
        {"title": "批量诊断", "icon": "🗂️", "callable": batch_diagnosis.main, "url_path": "batch"},
        {"title": "诊断结果", "icon": "📄", "callable": report_view.main, "url_path": "report"},
    ]


def build_global_style() -> str:
    return """
    <style>
    #MainMenu, footer, header[data-testid="stHeader"] [data-testid="stToolbar"], [data-testid="stDecoration"] {
      visibility: hidden;
    }
    .stApp {
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: #14324a;
    }
    .block-container {
      max-width: 1200px;
      padding-top: 1.2rem;
      padding-bottom: 2rem;
    }
    h1, h2, h3 {
      color: #0e3a5b;
    }
    .app-card {
      border: 1px solid #d9e4ec;
      border-radius: 14px;
      padding: 1rem 1.1rem;
      background: #f8fbfd;
      margin-bottom: 0.8rem;
    }
    .metric-card {
      border: 1px solid #c8d8e6;
      border-radius: 14px;
      padding: 0.9rem 1rem;
      background: white;
      min-height: 110px;
    }
    .metric-card .label {
      color: #53708a;
      font-size: 0.92rem;
      margin-bottom: 0.4rem;
    }
    .metric-card .value {
      color: #0e3a5b;
      font-size: 1.35rem;
      font-weight: 700;
    }
    div[data-testid="stMetric"] {
      background: white;
      border: 1px solid #c8d8e6;
      border-radius: 14px;
      padding: 0.8rem;
    }
    .home-hero {
      background: linear-gradient(135deg, #f3f9fd 0%, #ffffff 72%);
      border: 1px solid #dce9f2;
      border-radius: 18px;
      padding: 1.4rem 1.5rem;
      margin-bottom: 1.25rem;
    }
    .home-hero .eyebrow {
      color: #357093;
      font-size: 0.86rem;
      font-weight: 700;
      letter-spacing: 0.08em;
    }
    .home-hero h1 { color: #103954; margin: 0.28rem 0 0.32rem; font-size: 2.45rem; }
    .home-hero .subtitle { color: #647d90; font-size: 1.02rem; }
    .capability-card {
      border: 1px solid #d7e5ee;
      border-radius: 14px;
      padding: 0.85rem 0.95rem;
      background: #ffffff;
      min-height: 112px;
      margin-bottom: 0.65rem;
    }
    .capability-card .title { color: #163f5e; font-weight: 700; font-size: 1.02rem; }
    .capability-card .description { color: #61798c; font-size: 0.84rem; margin-top: 0.4rem; }
    .capability-card.normal { border-top: 4px solid #2f855a; }
    .capability-card.fault { border-top: 4px solid #cf4a4a; }
    .capability-card.warning { border-top: 4px solid #d39a28; }
    .flow-lane {
      display: flex;
      flex-direction: column;
      gap: 0.32rem;
      align-items: stretch;
      padding: 1rem;
      border-radius: 14px;
      background: #f8fbfd;
      border: 1px solid #d9e7f0;
    }
    .flow-step {
      padding: 0.52rem 0.68rem;
      border-radius: 10px;
      background: #ffffff;
      border: 1px solid #cfe0ed;
      color: #1f4663;
      font-size: 0.86rem;
      font-weight: 600;
    }
    .flow-arrow { color: #6f91aa; font-weight: 700; }
    .diagnosis-highlight {
      border-radius: 16px;
      padding: 1.15rem 1.25rem;
      margin: 0.35rem 0 1rem 0;
      border: 1px solid #d5e3ed;
    }
    .diagnosis-highlight .label { font-size: 0.9rem; font-weight: 700; }
    .diagnosis-highlight .result { font-size: 2rem; font-weight: 800; margin: 0.25rem 0; }
    .diagnosis-highlight .meta { font-size: 0.92rem; }
    .diagnosis-highlight.normal { background: #f0faf4; border-color: #b9dfc7; color: #276749; }
    .diagnosis-highlight.fault { background: #fff4f3; border-color: #f2c4c0; color: #a43838; }
    .diagnosis-highlight.warning { background: #fff9e9; border-color: #f0d694; color: #946a16; }
    .section-intro { color: #668095; margin-top: -0.25rem; margin-bottom: 0.75rem; }
    .input-note { color: #557187; background: #f4f9fc; border-left: 3px solid #4b86ad; padding: 0.65rem 0.8rem; border-radius: 6px; }
    .task-stat-caption { color: #61798c; font-size: 0.84rem; }
    .top-navigation-label { color: #6b8396; font-size: 0.78rem; font-weight: 700; margin: 0.1rem 0 0.35rem; }
    div[data-testid="stMetric"] { box-shadow: 0 2px 8px rgba(26, 70, 99, 0.04); }
    @media (max-width: 800px) {
      .home-hero { padding: 1rem; }
      .home-hero h1 { font-size: 2rem; }
      .diagnosis-highlight .result { font-size: 1.6rem; }
    }
    </style>
    """


def _st():
    import streamlit as st

    return st


def render_home_page(single_page: Any) -> None:
    st = _st()
    sections = build_home_sections()
    st.markdown(
        f'<div class="home-hero"><div class="eyebrow">VIBRATION INTELLIGENCE PLATFORM</div><h1>{sections["title"]}</h1><div class="subtitle">{sections["subtitle"]}</div></div>',
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
                f'<div class="app-card"><strong>{label}</strong><br>{value}</div>',
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
                f'{item["icon"]} {item["title"]}',
                key=f'top-nav-{item["url_path"]}',
                use_container_width=True,
            ):
                st.switch_page(page_by_path[item["url_path"]])


def main() -> None:
    st = _st()
    st.set_page_config(page_title=APP_TITLE, page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
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
