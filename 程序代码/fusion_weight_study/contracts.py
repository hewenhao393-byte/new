from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when an experiment input drifts from the frozen contract."""


CHANNELS = (3, 4, 5)
LABELS = ("正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀")
FEATURE_NAMES = (
    "rms", "std", "peak_to_peak", "skewness", "kurtosis", "crest_factor",
    "impulse_factor", "clearance_factor", "shape_factor", "rot_1x_energy_ratio",
    "rot_2x_energy_ratio", "rot_3x_energy_ratio", "rot_2x_1x_ratio",
    "rot_3x_1x_ratio", "harmonic_energy_ratio_1x_5x",
    "harmonic_energy_ratio_3x_5x", "rot_2x_harmonic_ratio",
    "rot_05x_1x_ratio", "noninteger_harmonic_energy_ratio", "spectral_entropy",
    "spectral_flatness", "spectral_centroid", "spectral_bandwidth",
    "band_energy_5_300_ratio", "band_energy_300_1000_ratio",
    "band_energy_1000_3000_ratio", "band_energy_3000_5000_ratio",
    "high_low_energy_ratio", "wp_energy_ratio_0", "wp_energy_ratio_1",
    "wp_energy_ratio_2", "wp_energy_ratio_3", "wp_energy_ratio_4",
    "wp_energy_ratio_5", "wp_energy_ratio_6", "wp_energy_ratio_7",
    "wp_energy_entropy", "env_kurtosis", "env_crest_factor",
    "env_spectral_entropy", "env_peak_energy_ratio", "env_peak_concentration",
    "env_peak_count",
)
MODEL_PARAMS = {
    "loss_function": "MultiClass", "iterations": 500, "depth": 8,
    "learning_rate": 0.05, "l2_leaf_reg": 100, "random_strength": 5,
    "rsm": 0.7, "auto_class_weights": "SqrtBalanced", "random_seed": 2026,
    "allow_writing_files": False, "verbose": False, "thread_count": 4,
}
CV_FOLDS = 3
PRIMARY_SELECTION_METRIC = "record_macro_f1"
SIMPLICITY_TOLERANCE = 0.005
PARENT_IDENTITY = "record_id.rsplit('::', 1)[0]"


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ContractError(f"source manifest {name} drift: {actual!r} != {expected!r}")


def validate_source_manifest(path: Path | str) -> Mapping[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _require_equal("labels", tuple(manifest.get("labels", ())), LABELS)
    _require_equal("features", tuple(manifest.get("features", ())), FEATURE_NAMES)
    _require_equal("params", manifest.get("params"), MODEL_PARAMS)
    _require_equal("cv_splits", manifest.get("cv_splits"), CV_FOLDS)
    _require_equal("split_mode", manifest.get("split_mode"), "parent_group")
    _require_equal("parent_identity", manifest.get("parent_identity"), PARENT_IDENTITY)
    return manifest
