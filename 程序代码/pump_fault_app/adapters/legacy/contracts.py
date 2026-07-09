from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyFeatureTableContract:
    source_package: str = "pump_diagnosis"
    expected_label_count: int = 6
    expected_feature_count: int = 21
    description: str = "Reserved adapter boundary for importing legacy six-class feature tables. Legacy adapters are for experiment traceability only and must not be used as formal software inference entrypoints."
