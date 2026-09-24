from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pump_fault_app.history.deletion import delete_history_record
from pump_fault_app.history.repository import SQLiteDiagnosisHistoryRepository
from pump_fault_app.history.service import (
    DEFAULT_HISTORY_DATABASE_PATH,
    DEFAULT_HISTORY_REPORT_DIR,
)
from pump_fault_app.presentation.history import (
    build_history_detail,
    build_history_select_options,
    build_history_summary,
    build_history_table_rows,
)
from pump_fault_app.ui.branding import build_page_header


def _st():
    import streamlit as st

    return st


def load_history_records(
    database_path: str | Path = DEFAULT_HISTORY_DATABASE_PATH,
):
    return SQLiteDiagnosisHistoryRepository(database_path).list_all()


def main() -> None:
    st = _st()
    st.markdown(
        build_page_header(
            title="历史诊断记录",
            subtitle="查看单文件诊断结果，并下载对应的Word诊断报告。",
        ),
        unsafe_allow_html=True,
    )
    notice = st.session_state.pop("history-delete-notice", None)
    if notice:
        st.success(notice)

    try:
        records = load_history_records()
    except Exception:
        st.error("历史记录暂时无法读取，请稍后重试。")
        return

    summary = build_history_summary(records)
    summary_columns = st.columns(2, gap="large")
    for column, (label, value) in zip(summary_columns, summary.items()):
        column.metric(label, value)

    if not records:
        st.info("暂无历史诊断记录，请先完成一次单文件诊断。")
        return

    st.markdown("### 历史结果")
    column_widths = [1.3, 2.4, 1.45, 0.85, 0.8, 2.0]
    header_columns = st.columns(column_widths, gap="small")
    for column, label in zip(
        header_columns,
        ["诊断时间", "文件名", "采样率 · 转速", "预测类别", "置信度", "操作"],
    ):
        column.caption(label)

    for record in records:
        table_row = build_history_table_rows([record])[0]
        with st.container(border=True):
            row_columns = st.columns(column_widths, gap="small")
            row_columns[0].write(table_row["诊断时间"])
            row_columns[1].markdown(
                "<span style='display: block; white-space: nowrap; "
                "overflow: hidden; text-overflow: ellipsis;' "
                f"title='{escape(record.file_name)}'>{escape(record.file_name)}</span>",
                unsafe_allow_html=True,
            )
            row_columns[2].markdown(
                "<span style='white-space: nowrap;'>"
                f"{record.sampling_rate_hz} Hz · {record.rpm:g} rpm"
                "</span>",
                unsafe_allow_html=True,
            )
            row_columns[3].write(table_row["预测类别"])
            row_columns[4].write(table_row["置信度"])

            confirmation_key = f"history-delete-confirm-{record.id}"
            with row_columns[5]:
                download_column, delete_column = st.columns(2, gap="small")
                with download_column:
                    if record.report_path.is_file():
                        st.download_button(
                            "下载",
                            data=record.report_path.read_bytes(),
                            file_name=record.report_path.name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"history-download-{record.id}",
                            width="stretch",
                        )
                    else:
                        st.button("报告缺失", disabled=True, width="stretch")

                with delete_column:
                    if not st.session_state.get(confirmation_key, False):
                        if st.button(
                            "删除",
                            key=f"history-delete-start-{record.id}",
                            width="stretch",
                        ):
                            st.session_state[confirmation_key] = True
                            st.rerun()

                if st.session_state.get(confirmation_key, False):
                    st.caption("确认后将同时删除该记录和Word报告。")
                    confirm_column, cancel_column = st.columns(2, gap="small")
                    with confirm_column:
                        confirmed = st.button(
                            "确认删除",
                            type="primary",
                            key=f"history-delete-confirm-button-{record.id}",
                            width="stretch",
                        )
                    with cancel_column:
                        cancelled = st.button(
                            "取消",
                            key=f"history-delete-cancel-{record.id}",
                            width="stretch",
                        )
                    if cancelled:
                        st.session_state.pop(confirmation_key, None)
                        st.rerun()
                    if confirmed:
                        repository = SQLiteDiagnosisHistoryRepository(
                            DEFAULT_HISTORY_DATABASE_PATH
                        )
                        deletion_result = delete_history_record(
                            repository,
                            record,
                            report_root=DEFAULT_HISTORY_REPORT_DIR,
                        )
                        messages = {
                            "deleted": "已删除历史记录及关联Word报告。",
                            "record_deleted_report_missing": "报告文件已不存在，已删除历史记录。",
                            "record_not_found": "历史记录不存在或已被删除。",
                            "report_delete_failed": "报告删除失败，历史记录未删除。",
                            "report_path_outside_system_directory": "报告路径不在系统报告目录内，未执行删除。",
                            "report_deleted_record_delete_failed": "报告已删除，但历史记录未删除，请刷新后重试。",
                        }
                        st.session_state.pop(confirmation_key, None)
                        if deletion_result.status in {
                            "deleted",
                            "record_deleted_report_missing",
                        }:
                            st.session_state["history-delete-notice"] = messages[
                                deletion_result.status
                            ]
                            st.rerun()
                        st.error(messages[deletion_result.status])

    st.markdown("### 记录详情")
    select_options = build_history_select_options(records)
    selected_id = st.selectbox(
        "选择需要查看详情的历史记录",
        options=list(select_options),
        format_func=select_options.get,
    )
    selected_record = next(
        record for record in records if record.id == selected_id
    )
    st.dataframe(
        [
            {"项目": label, "内容": value}
            for label, value in build_history_detail(selected_record).items()
        ],
        width="stretch",
        hide_index=True,
    )
