from __future__ import annotations

from pathlib import Path

from pump_fault_app.config.defaults import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    ENVELOPE_BAND_HIGH_HZ,
    ENVELOPE_BAND_LOW_HZ,
    FINAL_FEATURE_NAMES,
    SIX_CLASS_LABELS,
    TARGET_SAMPLE_RATE_HZ,
    WAVELET_LEVEL,
    WAVELET_NAME,
    WINDOW_SIZE,
    WINDOW_STEP,
    default_runtime_root,
)
from pump_fault_app.config.schema import AppConfig, FeatureConfig, LabelConfig, LoggingConfig, PathConfig, SignalConfig


def build_default_config(project_root: Path | None = None) -> AppConfig:
    resolved_root = Path.cwd() if project_root is None else Path(project_root)
    return AppConfig(
        signal=SignalConfig(
            target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
            bandpass_low_hz=BANDPASS_LOW_HZ,
            bandpass_high_hz=BANDPASS_HIGH_HZ,
            window_size=WINDOW_SIZE,
            window_step=WINDOW_STEP,
            envelope_band_low_hz=ENVELOPE_BAND_LOW_HZ,
            envelope_band_high_hz=ENVELOPE_BAND_HIGH_HZ,
            wavelet_name=WAVELET_NAME,
            wavelet_level=WAVELET_LEVEL,
        ),
        labels=LabelConfig(class_names=SIX_CLASS_LABELS),
        features=FeatureConfig(final_feature_names=FINAL_FEATURE_NAMES),
        paths=PathConfig(
            project_root=resolved_root,
            runtime_root=default_runtime_root(resolved_root),
        ),
        logging=LoggingConfig(
            logger_name="pump_fault_app",
            log_level="INFO",
            log_filename="pump_fault_app.log",
        ),
    )
