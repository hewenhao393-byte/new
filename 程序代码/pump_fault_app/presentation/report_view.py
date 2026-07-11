"""Display-only adapters for the diagnosis result page."""

from __future__ import annotations


def build_diagnosis_highlight(label: str | None, confidence: str, grade: str | None) -> dict[str, str]:
    display_label = label or "-"
    if display_label == "正常":
        tone, status = "normal", "正常"
    elif display_label == "-":
        tone, status = "warning", "建议复测"
    else:
        tone, status = "fault", "异常"
    return {
        "tone": tone,
        "status": status,
        "label": display_label,
        "confidence": confidence,
        "grade": grade or "-",
    }
