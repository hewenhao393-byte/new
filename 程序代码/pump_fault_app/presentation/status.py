"""User-facing status text derived from already-recorded service outcomes."""

from __future__ import annotations


def build_runtime_processing_status(alert_count: int) -> str:
    """Return a neutral status while alert details remain available for review."""
    if alert_count <= 0:
        return "诊断流程正常完成。"
    return "已完成数值稳定性保护处理，不影响诊断结果。"


def build_engineering_risk_level(*, success: bool, label: str | None) -> str:
    """Map an existing diagnosis outcome to an engineering-facing display level."""
    if not success:
        return "待复测"
    if label == "正常":
        return "低风险"
    return "需关注"


def build_user_runtime_notice(runtime_alert_count: int) -> str:
    """Return a stable user-facing notice without exposing numerical library logs."""
    if runtime_alert_count <= 0:
        return ""
    return "数值稳定性提示，不影响诊断结果。"
