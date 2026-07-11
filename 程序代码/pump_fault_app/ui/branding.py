"""Shared light scientific branding for Streamlit pages."""

from __future__ import annotations

from html import escape


APP_TITLE = "水泵智能故障诊断系统"
APP_SUBTITLE = "基于振动信号与机器学习的六分类故障识别"


def build_research_style() -> str:
    """Return the global CSS without coupling branding to Streamlit runtime state."""
    return """
    <style>
    #MainMenu, footer, header[data-testid="stHeader"] [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
    .stApp {
      background: #f7f9fc;
      color: #243b53;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .block-container { max-width: 1240px; padding-top: 1.1rem; padding-bottom: 2rem; }
    h1, h2, h3, h4 { color: #123b5d; }
    p, label, [data-testid="stMarkdownContainer"] { color: #334e68; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d9e2ec; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] { color: #486581; }
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover { background: #edf5fb; color: #0b6fa4; }
    [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stMetric"],
    [data-testid="stDataFrame"] {
      background: #ffffff;
      border-color: #d9e2ec;
      box-shadow: 0 2px 8px rgba(43, 87, 120, 0.04);
    }
    div[data-testid="stMetric"] { border-radius: 10px; padding: 0.9rem; }
    [data-testid="stMetricLabel"] { color: #627d98; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #123b5d; }
    [data-testid="stFileUploader"] section,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      background: #ffffff !important;
      color: #243b53 !important;
      border-color: #cbd5e1 !important;
    }
    .stButton > button { border-radius: 7px; border-color: #b8d4e6; background: #ffffff; color: #176b9a; font-weight: 600; }
    .stButton > button:hover { background: #edf7fc; border-color: #4b9bc4; color: #0b5f8d; }
    .stButton > button[kind="primary"] { background: #1778aa; border-color: #1778aa; color: #ffffff; }
    .research-hero { background: #ffffff; border: 1px solid #d9e6ee; border-left: 4px solid #2d88b9; border-radius: 12px; padding: 1.4rem 1.55rem; margin: 0.2rem 0 1.15rem; }
    .research-hero h1 { color: #123b5d; font-size: 2.2rem; margin: 0 0 0.35rem; }
    .research-hero .subtitle { color: #627d98; }
    .research-page-header { background: #ffffff; border: 1px solid #d9e6ee; border-left: 4px solid #2d88b9; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.15rem 0 1.05rem; }
    .research-page-header .title { color: #123b5d; font-size: 1.65rem; font-weight: 700; }
    .research-page-header .subtitle { color: #627d98; margin-top: 0.3rem; }
    .capability-card, .app-card, .research-card { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 10px; box-shadow: 0 2px 8px rgba(43, 87, 120, 0.04); }
    .research-card { padding: 1rem 1.1rem; margin: 0.5rem 0 1rem; }
    .capability-card { min-height: 105px; padding: 0.85rem 0.95rem; margin-bottom: 0.65rem; }
    .capability-card .title { color: #1d4e6d; font-weight: 700; }
    .capability-card .description, .section-intro, .task-stat-caption { color: #627d98; }
    .capability-card.normal { border-top: 4px solid #2f855a; }
    .capability-card.fault { border-top: 4px solid #d64545; }
    .capability-card.warning { border-top: 4px solid #d39a28; }
    .flow-lane { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.55rem; padding: 0.9rem; background: #ffffff; border: 1px solid #d9e2ec; border-radius: 10px; }
    .flow-step { background: #f2f8fc; color: #1d4e6d; padding: 0.62rem 0.72rem; border-left: 3px solid #4b9bc4; border-radius: 5px; font-size: 0.86rem; }
    .flow-arrow { display: none; }
    .diagnosis-highlight { border-radius: 10px; padding: 1.05rem 1.2rem; border: 1px solid #d9e2ec; background: #ffffff; }
    .diagnosis-highlight .label { color: #627d98; font-size: 0.82rem; font-weight: 700; }
    .diagnosis-highlight .result { color: #123b5d; font-size: 2rem; font-weight: 800; margin: 0.25rem 0; }
    .diagnosis-highlight .meta { color: #486581; }
    .diagnosis-highlight.normal { border-left: 4px solid #2f855a; }
    .diagnosis-highlight.fault { border-left: 4px solid #d64545; }
    .diagnosis-highlight.warning { border-left: 4px solid #d39a28; }
    .input-note { background: #edf7fc; border-left: 3px solid #4b9bc4; color: #486581; padding: 0.72rem 0.85rem; border-radius: 5px; }
    .top-navigation-label { color: #627d98; font-size: 0.78rem; font-weight: 700; margin: 0.25rem 0 0.35rem; }
    @media (max-width: 800px) {
      .flow-lane { grid-template-columns: 1fr; }
      .research-hero h1 { font-size: 1.7rem; }
    }
    </style>
    """


def build_page_header(*, title: str, subtitle: str) -> str:
    return (
        '<div class="research-page-header">'
        f'<div class="title">{escape(title)}</div>'
        f'<div class="subtitle">{escape(subtitle)}</div>'
        '</div>'
    )
