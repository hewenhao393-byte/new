from __future__ import annotations

from pathlib import Path

from pump_fault_app.app.bootstrap import bootstrap_application
from pump_fault_app.config.defaults import FINAL_FEATURE_NAMES, SIX_CLASS_LABELS
from pump_fault_app.config.loader import build_default_config
from pump_fault_app.domain.features import FEATURE_GROUPS


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
