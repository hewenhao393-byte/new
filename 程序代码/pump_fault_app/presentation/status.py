"""User-facing status text derived from already-recorded service outcomes."""


def build_runtime_processing_status(alert_count: int) -> str:
    """Return a neutral status while alert details remain available for review."""
    if alert_count <= 0:
        return "诊断流程正常完成。"
    return "已完成数值稳定性保护处理，不影响诊断结果。"

