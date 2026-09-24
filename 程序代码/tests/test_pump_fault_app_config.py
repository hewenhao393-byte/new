from pathlib import Path

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER, FORMAL_V3_CONTRACT
from pump_fault_app.prediction import DEFAULT_MODEL_DIRECTORY
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


def test_application_uses_only_v3_contract_and_deployment_directory() -> None:
    assert APP_VERSION == "pump-fault-app-v3"
    assert MODEL_VERSION == "catboost43-six-class-v3"
    assert FEATURE_VERSION == "43-feature-v1"
    assert INFERENCE_CONTRACT_VERSION == "formal-V3-catboost43"
    assert FORMAL_V3_CONTRACT.target_sampling_rate == 12_000
    assert FORMAL_V3_CONTRACT.filter_low_hz == 5.0
    assert FORMAL_V3_CONTRACT.filter_high_hz == 5_000.0
    assert FORMAL_V3_CONTRACT.window_size == 4_800
    assert FORMAL_V3_CONTRACT.step_size == 2_400
    assert len(FORMAL_FEATURE_NAMES) == 43
    assert len(FORMAL_LABEL_ORDER) == 6
    assert DEFAULT_MODEL_DIRECTORY.name == "catboost43_v3"
    assert (DEFAULT_MODEL_DIRECTORY / "manifest.json").is_file()
