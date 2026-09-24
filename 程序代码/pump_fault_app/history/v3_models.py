from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V3HistoryRecord:
    diagnosed_at: str
    source_files: dict[str, str]
    input_channels: tuple[str, ...]
    sampling_rate_hz: int
    rpm: float
    status: str
    predicted_label: str | None
    fused_probabilities: tuple[float, ...] | None
    channel_results: tuple[dict[str, Any], ...]
    agreement_level: str | None
    agreement_count: int
    model_version: str
    feature_version: str
    contract_version: str
    warnings: tuple[str, ...]
    report_path: Path | None = None
    id: int | None = None
