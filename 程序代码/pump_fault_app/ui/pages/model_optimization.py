from __future__ import annotations

from html import escape
from io import StringIO

import pandas as pd

from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.history.repository import SQLiteDiagnosisHistoryRepository
from pump_fault_app.history.service import DEFAULT_HISTORY_DATABASE_PATH
from pump_fault_app.model_registry import (
    DEFAULT_CANDIDATE_MODEL_DIR,
    DEFAULT_MODEL_REGISTRY_PATH,
    SQLiteModelRegistry,
)
from pump_fault_app.model_training import train_candidate_model
from pump_fault_app.sample_repository import (
    DEFAULT_SAMPLE_DATABASE_PATH,
    SQLiteSampleRepository,
)
from pump_fault_app.ui.branding import build_page_header


def _st():
    import streamlit as st

    return st


def _as_csv(rows: list[dict[str, object]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def main() -> None:
    st = _st()
    st.markdown(
        build_page_header(
            title="模型持续优化",
            subtitle="基于人工确认样本构建候选 BP 模型；候选版本需人工批准，且不会自动替换正式诊断模型。",
        ),
        unsafe_allow_html=True,
    )
    st.info(
        "正式诊断仍固定使用当前冻结 BP 模型、21维特征、六分类标签和预处理参数。"
    )
    history_repository = SQLiteDiagnosisHistoryRepository(DEFAULT_HISTORY_DATABASE_PATH)
    sample_repository = SQLiteSampleRepository(DEFAULT_SAMPLE_DATABASE_PATH)
    registry = SQLiteModelRegistry(DEFAULT_MODEL_REGISTRY_PATH)
    records = history_repository.list_all()

    st.markdown("### 1. 历史样本人工确认")
    st.caption("确认类别以设备检修或现场复核结果为准；只有具备21维特征快照的记录可进入训练数据集。")
    if not records:
        st.info("暂无历史诊断记录。请先在“单文件诊断”中完成一次成功诊断。")
    else:
        record_ids = tuple(int(record.id) for record in records)
        record_option_labels = {
            int(record.id): f"{record.diagnosed_at} · {record.file_name} · {record.predicted_label}"
            for record in records
        }
        selected_record_id = st.selectbox(
            "选择需要人工确认的历史样本",
            record_ids,
            format_func=record_option_labels.get,
            key="optimization-history-record",
        )
        selected_index = record_ids.index(selected_record_id)
        record = records[selected_index]
        st.caption(f"当前第 {selected_index + 1} / {len(records)} 条")
        annotation = sample_repository.get_annotation(int(record.id))
        snapshot_count = sample_repository.snapshot_count(history_record_id=int(record.id))
        with st.container(border=True):
            header = st.columns([2.2, 2.8, 1.2, 1.2, 1.5], gap="small")
            for column, text in zip(header, ("诊断时间", "文件名", "预测类别", "特征快照", "真实类别确认")):
                column.caption(text)
            columns = st.columns([2.2, 2.8, 1.2, 1.2, 1.5], gap="small")
            columns[0].write(record.diagnosed_at)
            columns[1].write(record.file_name)
            columns[2].write(record.predicted_label)
            columns[3].write(f"{snapshot_count} 个窗口" if snapshot_count else "未保存")
            with columns[4]:
                selected = st.selectbox(
                    "真实类别",
                    FORMAL_LABEL_ORDER,
                    index=FORMAL_LABEL_ORDER.index(annotation.true_label)
                    if annotation is not None
                    else FORMAL_LABEL_ORDER.index(record.predicted_label),
                    key=f"optimization-label-{record.id}",
                    label_visibility="collapsed",
                )
                if st.button("保存确认", key=f"optimization-save-{record.id}", width="stretch"):
                    sample_repository.save_annotation(
                        history_record_id=int(record.id),
                        true_label=selected,
                    )
                    st.rerun()
            if snapshot_count == 0:
                st.caption("该旧记录缺少特征快照，可保存人工确认，但不会纳入训练数据集。")

    st.markdown("### 2. 训练数据集与候选模型")
    training_rows = sample_repository.list_training_rows()
    group_count = len({int(row["history_record_id"]) for row in training_rows})
    metric_columns = st.columns(3, gap="large")
    metric_columns[0].metric("已确认窗口样本", len(training_rows))
    metric_columns[1].metric("独立历史样本", group_count)
    metric_columns[2].metric("特征维度", "21")
    if training_rows:
        st.download_button(
            "导出训练数据集 CSV",
            data=_as_csv(training_rows),
            file_name="pump_fault_confirmed_training_dataset.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("尚无可导出的已确认特征样本。")

    label_counts = pd.Series(
        [str(row["true_label"]) for row in training_rows], dtype="object"
    ).value_counts()
    missing_labels = [label for label in FORMAL_LABEL_ORDER if label not in label_counts]
    insufficient_labels = [
        label
        for label in FORMAL_LABEL_ORDER
        if sum(
            1
            for record_id in {
                int(row["history_record_id"])
                for row in training_rows
                if row["true_label"] == label
            }
        ) < 5
    ]
    if missing_labels or insufficient_labels:
        st.warning(
            "候选训练条件未满足：每个六分类类别至少需要5条独立历史记录。"
            f" 当前缺少或不足类别：{'、'.join(sorted(set(missing_labels + insufficient_labels))) or '无'}。"
        )
    elif st.button("训练候选 BP 模型", type="primary", width="stretch"):
        try:
            training_result = train_candidate_model(
                training_rows,
                output_dir=DEFAULT_CANDIDATE_MODEL_DIR,
            )
            registry.add_candidate(
                version=training_result.version,
                bundle_path=training_result.bundle_path,
                trained_at=training_result.trained_at,
                sample_count=training_result.sample_count,
                accuracy=training_result.accuracy,
                macro_f1=training_result.macro_f1,
            )
            st.success("候选模型已完成训练并登记，等待人工批准。")
            st.rerun()
        except ValueError as exc:
            st.error(f"候选模型未生成：{exc}")

    st.markdown("### 3. 模型版本与人工批准")
    formal = registry.current_formal_version()
    st.markdown(
        """
        <style>
        .model-version-card {
            min-height: 8.25rem;
            padding: 1rem 1.1rem;
        }
        .model-version-label {
            color: #627d98;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.65rem;
        }
        .model-version-value {
            color: #123b5d;
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.45;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    current_columns = st.columns((1.15, 1.15, 0.7), gap="large")
    with current_columns[0]:
        st.markdown(
            '<div class="app-card model-version-card">'
            '<div class="model-version-label">当前正式模型版本</div>'
            f'<div class="model-version-value">{escape(formal.version)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with current_columns[1]:
        st.markdown(
            '<div class="app-card model-version-card">'
            '<div class="model-version-label">正式模型状态</div>'
            f'<div class="model-version-value">{escape(formal.status)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    current_columns[2].metric("候选版本数量", len(registry.list_candidates()))

    candidates = registry.list_candidates()
    if not candidates:
        st.caption("尚未生成候选模型。")
        return
    candidate_rows = [
        {
            "版本": item.version,
            "训练时间": item.trained_at,
            "样本数量": item.sample_count,
            "测试准确率": f"{item.accuracy:.1%}",
            "Macro-F1": f"{item.macro_f1:.1%}",
            "状态": item.status,
        }
        for item in candidates
    ]
    st.dataframe(candidate_rows, width="stretch", hide_index=True)
    pending = [item for item in candidates if item.status == "待人工批准"]
    if pending:
        selected_version = st.selectbox(
            "选择待批准候选版本",
            [item.version for item in pending],
        )
        if st.button("人工批准候选版本", width="stretch"):
            registry.approve(selected_version)
            st.success("候选版本已批准；正式诊断模型未被替换。")
            st.rerun()
