from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelVersionRecord:
    version: str
    bundle_path: Path | None
    trained_at: str
    sample_count: int
    accuracy: float | None
    macro_f1: float | None
    status: str
