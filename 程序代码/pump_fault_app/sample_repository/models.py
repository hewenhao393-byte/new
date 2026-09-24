from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfirmedAnnotation:
    history_record_id: int
    true_label: str
    confirmed_at: str


@dataclass(frozen=True)
class FeatureSnapshot:
    history_record_id: int
    window_index: int
    feature_values: tuple[float, ...]
