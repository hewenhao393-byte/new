from __future__ import annotations

from typing import Any

from pump_fault_app.services import AppBatchRunResult, AppSingleRunResult
from pump_fault_app.ui.streamlit_app import APP_SUBTITLE, APP_TITLE, build_home_sections


def build_report_overview(single_result: AppSingleRunResult | None, batch_result: AppBatchRunResult | None) -> dict[str, Any]:
    return {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "single_available": single_result is not None,
        "batch_available": batch_result is not None,
    }


def _st():
    import streamlit as st

    return st


def main() -> None:
    st = _st()
    overview = build_report_overview(
        st.session_state.get("single_run_result"),
        st.session_state.get("batch_run_result"),
    )
    st.title("报告查看")
    st.caption(overview["subtitle"])
    st.markdown("### 系统首页信息")
    st.json(build_home_sections())
    st.markdown("### 当前缓存状态")
    st.json(
        {
            "single_available": overview["single_available"],
            "batch_available": overview["batch_available"],
        }
    )
    if overview["single_available"]:
        st.markdown("### 最近一次单文件摘要")
        st.json(st.session_state["single_run_result"].summary.as_dict())
    if overview["batch_available"]:
        st.markdown("### 最近一次批量统计")
        st.json(
            {
                "total_count": st.session_state["batch_run_result"].batch_result.total_count,
                "diagnosed_count": st.session_state["batch_run_result"].batch_result.diagnosed_count,
                "rejected_count": st.session_state["batch_run_result"].batch_result.rejected_count,
            }
        )


if __name__ == "__main__":
    main()
