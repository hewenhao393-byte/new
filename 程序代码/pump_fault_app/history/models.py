from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagnosisHistoryRecord:
    diagnosed_at: str
    file_name: str
    sampling_rate_hz: int
    rpm: float
    predicted_label: str
    confidence: float
    window_consistency: float
    report_path: Path
    id: int | None = None
