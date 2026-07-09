from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from pump_diagnosis.inference_contract import FORMAL_MODEL_VERSION, FORMAL_SOFTWARE_VERSION

from pump_fault_app.reporting.view_data import SingleReportViewData


def export_single_report_to_docx(view_data: SingleReportViewData, output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    _configure_document(document)
    _add_title(document)
    _add_basic_info(document, view_data)
    _add_conclusion(document, view_data)
    _add_probability_table(document, view_data)
    _add_window_distribution_table(document, view_data)
    _add_visualizations(document, view_data)
    _add_scope_note(document)
    _add_generation_metadata(document)
    document.save(target_path)
    return target_path


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    normal_style = document.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10.5)


def _add_title(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("水泵振动故障诊断报告")
    run.bold = True
    run.font.size = Pt(18)
    document.add_paragraph("")


def _add_basic_info(document: Document, view_data: SingleReportViewData) -> None:
    document.add_heading("1. 基本信息", level=1)
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "内容"
    for item in view_data.basic_info:
        row = table.add_row().cells
        row[0].text = item.label
        row[1].text = item.value


def _add_conclusion(document: Document, view_data: SingleReportViewData) -> None:
    document.add_heading("2. 诊断结论", level=1)
    conclusion = view_data.to_dict()["conclusion"]
    rows = [
        ("最终诊断类别", conclusion["final_label"]),
        ("最高平均概率", conclusion["confidence_text"]),
        ("第二可能类别", conclusion["second_label"]),
        ("概率差", conclusion["probability_margin_text"]),
        ("窗口一致率", conclusion["window_consistency_text"]),
        ("诊断等级", conclusion["diagnosis_level"]),
        ("诊断建议", conclusion["suggestion"]),
    ]
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "内容"
    for index, (label, value) in enumerate(rows):
        row = table.add_row().cells
        row[0].text = str(label)
        paragraph = row[1].paragraphs[0]
        run = paragraph.add_run(str(value))
        if index in (0, 1):
            run.bold = True
    warnings = conclusion["warnings"]
    if warnings:
        document.add_paragraph("警告信息：")
        for warning in warnings:
            document.add_paragraph(str(warning), style="List Bullet")


def _add_probability_table(document: Document, view_data: SingleReportViewData) -> None:
    document.add_heading("3. 六类概率分布", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "类别"
    table.rows[0].cells[1].text = "平均概率"
    table.rows[0].cells[2].text = "百分比"
    for item in view_data.probabilities:
        row = table.add_row().cells
        row[0].text = item.label
        row[1].text = f"{item.probability:.3f}"
        row[2].text = item.percentage_text


def _add_window_distribution_table(document: Document, view_data: SingleReportViewData) -> None:
    document.add_heading("4. 窗口预测分布", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "类别"
    table.rows[0].cells[1].text = "窗口数量"
    table.rows[0].cells[2].text = "占比"
    for item in view_data.window_distribution:
        row = table.add_row().cells
        row[0].text = item.label
        row[1].text = str(item.window_count)
        row[2].text = f"{item.ratio:.1%}"


def _add_visualizations(document: Document, view_data: SingleReportViewData) -> None:
    document.add_heading("5. 信号分析图", level=1)
    availability = view_data.visualization_availability
    with tempfile.TemporaryDirectory(prefix="pump_report_figs_") as tmp_dir:
        temp_root = Path(tmp_dir)
        _add_optional_line_chart(
            document,
            title="时域波形",
            available=availability.time_domain,
            unavailable_message="该图形数据不可用：时域波形数据不可用",
            image_path=temp_root / "time_domain.png",
            x_values=None if view_data.time_domain is None else view_data.time_domain.time_s,
            y_values=None if view_data.time_domain is None else view_data.time_domain.amplitude,
            x_label="Time (s)",
            y_label="Amplitude",
            color="#0f5b8d",
        )
        _add_optional_line_chart(
            document,
            title="0–5000 Hz 频谱",
            available=availability.frequency_spectrum,
            unavailable_message="该图形数据不可用：频谱图数据不可用",
            image_path=temp_root / "frequency_spectrum.png",
            x_values=None if view_data.frequency_spectrum is None else view_data.frequency_spectrum.frequency_hz,
            y_values=None if view_data.frequency_spectrum is None else view_data.frequency_spectrum.amplitude,
            x_label="Frequency (Hz)",
            y_label="Amplitude",
            color="#157a6e",
        )
        _add_optional_line_chart(
            document,
            title="包络谱",
            available=availability.envelope_spectrum,
            unavailable_message="该图形数据不可用：包络谱数据不可用",
            image_path=temp_root / "envelope_spectrum.png",
            x_values=None if view_data.envelope_spectrum is None else view_data.envelope_spectrum.frequency_hz,
            y_values=None if view_data.envelope_spectrum is None else view_data.envelope_spectrum.amplitude,
            x_label="Frequency (Hz)",
            y_label="Amplitude",
            color="#a05a2c",
        )
        _add_optional_bar_chart(
            document,
            title="小波包能量占比",
            available=availability.wavelet_packet_energy,
            unavailable_message="该图形数据不可用：小波包能量图数据不可用",
            image_path=temp_root / "wavelet_packet.png",
            labels=None if view_data.wavelet_packet_energy is None else view_data.wavelet_packet_energy.band_labels,
            values=None if view_data.wavelet_packet_energy is None else view_data.wavelet_packet_energy.energy_ratio,
            x_label="Band",
            y_label="Energy Ratio",
            color="#2f6f4f",
        )

    if availability.messages:
        document.add_heading("6. 可视化缺失提示", level=1)
        for message in availability.messages:
            document.add_paragraph(str(message), style="List Bullet")


def _add_optional_line_chart(
    document: Document,
    *,
    title: str,
    available: bool,
    unavailable_message: str,
    image_path: Path,
    x_values,
    y_values,
    x_label: str,
    y_label: str,
    color: str,
) -> None:
    document.add_heading(title, level=2)
    if not available or x_values is None or y_values is None:
        document.add_paragraph(unavailable_message)
        return
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(list(x_values), list(y_values), color=color, linewidth=1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    document.add_picture(str(image_path), width=Inches(6.2))
    _add_figure_caption(document, title)


def _add_optional_bar_chart(
    document: Document,
    *,
    title: str,
    available: bool,
    unavailable_message: str,
    image_path: Path,
    labels,
    values,
    x_label: str,
    y_label: str,
    color: str,
) -> None:
    document.add_heading(title, level=2)
    if not available or labels is None or values is None:
        document.add_paragraph(unavailable_message)
        return
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.bar(list(labels), list(values), color=color)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    document.add_picture(str(image_path), width=Inches(6.2))
    _add_figure_caption(document, title)


def _add_scope_note(document: Document) -> None:
    document.add_heading("7. 说明与适用范围", level=1)
    document.add_paragraph(
        "本报告结果由已训练的 BP 神经网络六分类模型根据输入振动信号自动生成。"
        "诊断结论用于辅助状态判断和检修排查，不应替代现场专业检测。"
        "当前模型主要适用于与训练数据采集条件相近的水泵振动信号。"
    )


def _add_generation_metadata(document: Document) -> None:
    document.add_heading("8. 生成信息", level=1)
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "内容"
    rows = [
        ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("模型版本", FORMAL_MODEL_VERSION),
        ("软件版本", FORMAL_SOFTWARE_VERSION),
    ]
    for label, value in rows:
        row = table.add_row().cells
        row[0].text = label
        row[1].text = value


def _add_figure_caption(document: Document, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(f"图：{title}")
    run.italic = True
