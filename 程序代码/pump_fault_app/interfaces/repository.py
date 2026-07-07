from __future__ import annotations

from typing import Protocol

from pump_fault_app.domain.records import DiagnosisResult, FeatureVector, WindowSlice


class SampleRepository(Protocol):
    def save_window(self, window: WindowSlice) -> None:
        ...

    def save_feature_vector(self, vector: FeatureVector) -> None:
        ...

    def save_diagnosis_result(self, result: DiagnosisResult) -> None:
        ...
