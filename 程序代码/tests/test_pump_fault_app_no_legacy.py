from __future__ import annotations

from pathlib import Path


FORBIDDEN = (
    "FORMAL_V2_CONTRACT",
    "load_formal_bp_bundle",
    "StandardScaler",
    "MLPClassifier",
    "bp_bundle.joblib",
)


def test_deployed_app_has_no_v2_bp_references() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("pump_fault_app").rglob("*.py"))
    )
    for forbidden in FORBIDDEN:
        assert forbidden not in source, forbidden


def test_obsolete_application_paths_are_absent() -> None:
    obsolete = (
        "pump_fault_app/feature_extraction/extractor.py",
        "pump_fault_app/prediction/predictor.py",
        "pump_fault_app/fusion/probability_fusion.py",
        "pump_fault_app/inference/service.py",
        "pump_fault_app/model_training",
        "pump_fault_app/model_registry",
        "pump_fault_app/sample_repository",
        "pump_fault_app/ui/pages/model_optimization.py",
        "tests/test_pump_fault_app_model_optimization.py",
    )
    assert [path for item in obsolete if (path := Path(item)).exists()] == []
