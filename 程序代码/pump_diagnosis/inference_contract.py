"""Legacy experiment import path for the deployed formal V2 contract.

Do not use this module as a new application dependency. Formal software code
imports ``pump_fault_app.domain.formal_contract`` directly.
"""

from pump_fault_app.domain.formal_contract import (
    FORMAL_BUNDLE_KEYS,
    FORMAL_FEATURE_NAMES,
    FORMAL_LABEL_ORDER,
    FORMAL_MODEL_BUNDLE_PATH,
    FORMAL_MODEL_VERSION,
    FORMAL_SOFTWARE_VERSION,
    FORMAL_V2_CONTRACT,
    FeatureContract,
    FormalInferenceContract,
    LabelContract,
    ModelContract,
    SignalContract,
    SoftwareContract,
    validate_feature_names,
    validate_label_order,
    validate_model_bundle,
)

__all__ = [
    "FORMAL_BUNDLE_KEYS",
    "FORMAL_FEATURE_NAMES",
    "FORMAL_LABEL_ORDER",
    "FORMAL_MODEL_BUNDLE_PATH",
    "FORMAL_MODEL_VERSION",
    "FORMAL_SOFTWARE_VERSION",
    "FORMAL_V2_CONTRACT",
    "FeatureContract",
    "FormalInferenceContract",
    "LabelContract",
    "ModelContract",
    "SignalContract",
    "SoftwareContract",
    "validate_feature_names",
    "validate_label_order",
    "validate_model_bundle",
]
