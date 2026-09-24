"""Presentation adapters for formatting existing service results."""

from pump_fault_app.presentation.history import (
    build_history_detail,
    build_history_select_options,
    build_history_summary,
    build_history_table_rows,
)
from pump_fault_app.presentation.status import (
    build_engineering_risk_level,
    build_runtime_processing_status,
    build_user_runtime_notice,
)
from pump_fault_app.presentation.system_overview import build_system_overview

__all__ = [
    "build_engineering_risk_level",
    "build_history_detail",
    "build_history_select_options",
    "build_history_summary",
    "build_history_table_rows",
    "build_runtime_processing_status",
    "build_system_overview",
    "build_user_runtime_notice",
]
