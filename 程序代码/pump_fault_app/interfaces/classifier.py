from __future__ import annotations

from typing import Protocol

from pump_fault_app.config.schema import AppConfig
from pump_fault_app.domain.records import DiagnosisResult, FeatureVector


class Classifier(Protocol):
    def predict(self, vector: FeatureVector, config: AppConfig) -> DiagnosisResult:
        ...
