from __future__ import annotations

from pump_fault_app.config.defaults import SIX_CLASS_LABELS


def is_valid_label(label: str) -> bool:
    return label in SIX_CLASS_LABELS


def assert_valid_label(label: str) -> str:
    if not is_valid_label(label):
        raise ValueError(f"unsupported label: {label}")
    return label
