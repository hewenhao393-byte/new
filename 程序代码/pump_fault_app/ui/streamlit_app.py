from __future__ import annotations

from typing import Any

from pump_fault_app.ui.pages import batch_diagnosis, report_view, single_diagnosis

APP_TITLE = "水泵智能故障诊断系统"
APP_SUBTITLE = "基于振动信号与机器学习的六分类故障识别"


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
        "pipeline": (
            "文件上传",
            "信号质量检查",
            "预处理",
            "特征提取",
            "BP模型推理",
            "诊断报告",
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
        {"title": "诊断报告", "icon": "📄", "callable": report_view.main, "url_path": "report"},
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
    .fault-tag {
      display: inline-block;
      padding: 0.45rem 0.8rem;
      margin: 0.2rem 0.3rem 0.2rem 0;
      border-radius: 999px;
      background: #e9f4fb;
      color: #0e3a5b;
      border: 1px solid #c7deed;
      font-size: 0.92rem;
    }
    .flow-step {
      display: inline-block;
      padding: 0.5rem 0.75rem;
      margin: 0.2rem 0.3rem 0.2rem 0;
      border-radius: 10px;
      background: #f2f7fb;
      border: 1px solid #d7e4ef;
      color: #1f4663;
      font-size: 0.92rem;
    }
    </style>
    """


def _st():
    import streamlit as st

    return st


def render_home_page() -> None:
    st = _st()
    sections = build_home_sections()
    st.title(sections["title"])
    st.caption(sections["subtitle"])

    left, right = st.columns((1.6, 1.0), gap="large")
    with left:
        st.markdown("### 六类故障")
        st.markdown(
            "".join(f'<span class="fault-tag">{fault}</span>' for fault in sections["faults"]),
            unsafe_allow_html=True,
        )
        st.markdown("### 系统处理流程")
        st.markdown(
            " ".join(
                [f'<span class="flow-step">{step}</span>' for step in sections["pipeline"][:-1]]
                + [f'<span class="flow-step">{sections["pipeline"][-1]}</span>']
            ).replace("</span> <span", "</span> → <span"),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### 系统参数信息")
        for label, value in sections["parameters"].items():
            st.markdown(
                f'<div class="app-card"><strong>{label}</strong><br>{value}</div>',
                unsafe_allow_html=True,
            )

    if st.button("开始单文件诊断", type="primary", use_container_width=True):
        st.switch_page("pump_fault_app/ui/pages/single_diagnosis.py")


def main() -> None:
    st = _st()
    st.set_page_config(page_title=APP_TITLE, page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
    st.markdown(build_global_style(), unsafe_allow_html=True)
    pages = [
        st.Page(item["callable"], title=item["title"], icon=item["icon"], url_path=item["url_path"])
        for item in build_navigation_items()
    ]
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
