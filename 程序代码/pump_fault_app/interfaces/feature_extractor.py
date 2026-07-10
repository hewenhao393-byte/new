from __future__ import annotations

from typing import Protocol

from pump_fault_app.config.schema import AppConfig
from pump_fault_app.domain.records import FeatureVector, SignalWindow


class FeatureExtractor(Protocol):
    def extract(self, window: SignalWindow, config: AppConfig) -> FeatureVector:
        ...
