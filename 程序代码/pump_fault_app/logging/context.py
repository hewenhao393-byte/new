from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoggingContext:
    run_id: str
    component: str
    label_space: str = "six_class"
