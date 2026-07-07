from __future__ import annotations

from typing import Any

APP_TITLE = "水泵智能故障诊断系统"
APP_SUBTITLE = "基于振动信号与机器学习的六分类故障识别"


def build_home_sections() -> dict[str, Any]:
    return {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "modules": (
            "单文件诊断",
            "批量诊断",
            "报告查看",
        ),
        "notes": (
            "界面层仅调用 pump_fault_app.services 中的统一 service API。",
            "正式推理参数与标签顺序由后端契约统一控制。",
        ),
    }


def _st():
    import streamlit as st

    return st


def main() -> None:
    st = _st()
    sections = build_home_sections()
    st.set_page_config(page_title=APP_TITLE, page_icon="🛠️", layout="wide")
    st.title(sections["title"])
    st.caption(sections["subtitle"])
    st.markdown("### 功能模块")
    for module in sections["modules"]:
        st.markdown(f"- {module}")
    st.markdown("### 说明")
    for note in sections["notes"]:
        st.info(note)


if __name__ == "__main__":
    main()
