from __future__ import annotations

from pathlib import Path

from pump_fault_app.app.bootstrap import bootstrap_application
from pump_fault_app.config.defaults import FINAL_FEATURE_NAMES, SIX_CLASS_LABELS
from pump_fault_app.domain.formal_contract import FORMAL_V2_CONTRACT
from pump_fault_app.domain.formal_contract import FORMAL_MODEL_BUNDLE_PATH
from pump_fault_app.config.loader import build_default_config
from pump_fault_app.domain.features import FEATURE_GROUPS
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


def test_default_config_uses_fixed_signal_parameters(tmp_path: Path) -> None:
    config = build_default_config(project_root=tmp_path)

    assert config.signal.target_sample_rate_hz == 12_000
    assert config.signal.bandpass_low_hz == 10.0
    assert config.signal.bandpass_high_hz == 5000.0
    assert config.signal.window_size == 2400
    assert config.signal.window_step == 1200
    assert config.signal.envelope_band_low_hz == 2000.0
    assert config.signal.envelope_band_high_hz == 5000.0
    assert config.signal.wavelet_name == "db6"
    assert config.signal.wavelet_level == 3


def test_labels_and_final_feature_names_follow_existing_six_class_contract() -> None:
    assert SIX_CLASS_LABELS == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    )
    assert FINAL_FEATURE_NAMES == (
        "kurtosis",
        "skewness",
        "crest_factor",
        "impulse_factor",
        "clearance_factor",
        "shape_factor",
        "rot_2x_1x_ratio",
        "rot_3x_1x_ratio",
        "harmonic_energy_ratio_1x_5x",
        "spectral_entropy",
        "spectral_flatness",
        "wp_energy_ratio_0",
        "wp_energy_ratio_1",
        "wp_energy_ratio_2",
        "wp_energy_ratio_3",
        "wp_energy_ratio_4",
        "wp_energy_ratio_5",
        "wp_energy_ratio_6",
        "wp_energy_ratio_7",
        "env_kurtosis",
        "env_crest_factor",
    )
    assert tuple(name for group in FEATURE_GROUPS for name in group.feature_names) == FINAL_FEATURE_NAMES


def test_bootstrap_returns_summary_with_paths_and_feature_count(tmp_path: Path) -> None:
    state = bootstrap_application(project_root=tmp_path)

    assert state.summary["label_count"] == 6
    assert state.summary["feature_count"] == 21
    assert state.summary["target_sample_rate_hz"] == 12_000
    assert state.summary["log_directory"] == str(tmp_path / "runtime_logs")
    assert state.config.paths.runtime_root == tmp_path / "runtime_logs"


def test_application_owned_contract_preserves_frozen_v2_values() -> None:
    assert FORMAL_V2_CONTRACT.target_sampling_rate == 12_000
    assert FORMAL_V2_CONTRACT.filter_low_hz == 10.0
    assert FORMAL_V2_CONTRACT.filter_high_hz == 5000.0
    assert FORMAL_V2_CONTRACT.window_size == 2400
    assert FORMAL_V2_CONTRACT.step_size == 1200
    assert FORMAL_V2_CONTRACT.feature_names == FINAL_FEATURE_NAMES
    assert FORMAL_V2_CONTRACT.label_order == SIX_CLASS_LABELS


def test_application_versions_are_non_empty() -> None:
    assert APP_VERSION
    assert MODEL_VERSION
    assert FEATURE_VERSION
    assert INFERENCE_CONTRACT_VERSION


def test_application_contract_resolves_model_bundle_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    expected_path = (
        project_root
        / "实验结果"
        / "多转速统一六分类实验V2"
        / "six_class_models"
        / "bp"
        / "bp_bundle.joblib"
    )

    assert FORMAL_MODEL_BUNDLE_PATH == expected_path


def test_architecture_document_describes_layers_and_frozen_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    architecture_doc = project_root / "docs" / "pump_fault_app_architecture.md"
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    text = architecture_doc.read_text(encoding="utf-8")
    assert "UI层" in text
    assert "Service层" in text
    assert "Inference层" in text
    assert "12000 Hz" in text
    assert "2400/1200" in text
    assert "db6" in text
    assert "pump_fault_app_architecture.md" in readme
