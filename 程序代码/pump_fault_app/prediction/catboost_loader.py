from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "catboost43_v3"
CHANNELS = ("CH3", "CH4", "CH5")


@dataclass(frozen=True)
class LoadedCatBoost43Models:
    models: Mapping[str, Any]
    feature_names: tuple[str, ...]
    label_order: tuple[str, ...]
    manifest: Mapping[str, Any]
    model_directory: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(model_directory: Path) -> dict[str, Any]:
    manifest_path = model_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"deployment manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"deployment manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("deployment manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    expected_versions = {
        "software_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "contract_version": INFERENCE_CONTRACT_VERSION,
    }
    for field, expected in expected_versions.items():
        if manifest.get(field) != expected:
            raise ValueError(f"deployment manifest {field} mismatch")
    if manifest.get("training_scope") != "all_accepted_data":
        raise ValueError("deployment manifest training scope mismatch")
    if manifest.get("intended_use") != "software_inference_only":
        raise ValueError("deployment manifest intended use mismatch")
    if manifest.get("evaluation_claims") is not None:
        raise ValueError("deployment manifest must not contain evaluation claims")
    if manifest.get("parent_group_evaluation_reused") is not False:
        raise ValueError("deployment manifest parent_group boundary mismatch")
    if tuple(manifest.get("features", ())) != FORMAL_FEATURE_NAMES:
        raise ValueError("deployment manifest feature order mismatch")
    if tuple(manifest.get("labels", ())) != FORMAL_LABEL_ORDER:
        raise ValueError("deployment manifest label order mismatch")
    models = manifest.get("models")
    if not isinstance(models, dict) or tuple(models) != CHANNELS:
        raise ValueError("deployment manifest must declare CH3, CH4, CH5 in order")


def load_catboost43_models(model_directory: Path | str | None = None) -> LoadedCatBoost43Models:
    from catboost import CatBoostClassifier

    directory = Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    directory = directory.expanduser().resolve()
    manifest = _read_manifest(directory)
    _validate_manifest(manifest)

    loaded: dict[str, Any] = {}
    for channel in CHANNELS:
        entry = manifest["models"][channel]
        if not isinstance(entry, dict):
            raise ValueError(f"deployment manifest model entry is invalid: {channel}")
        file_name = entry.get("file")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError(f"deployment model file name is invalid: {channel}")
        model_path = directory / file_name
        if not model_path.is_file():
            raise FileNotFoundError(f"deployment model is missing: {model_path}")
        if _sha256_file(model_path) != entry.get("sha256"):
            raise ValueError(f"deployment model SHA-256 mismatch: {channel}")

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        if tuple(model.feature_names_) != FORMAL_FEATURE_NAMES:
            raise ValueError(f"deployment model feature order mismatch: {channel}")
        classes = tuple(str(label) for label in model.classes_)
        if len(classes) != len(FORMAL_LABEL_ORDER) or set(classes) != set(FORMAL_LABEL_ORDER):
            raise ValueError(f"deployment model class set mismatch: {channel}")
        if tuple(entry.get("classes", ())) != classes:
            raise ValueError(f"deployment manifest class order mismatch: {channel}")
        loaded[channel] = model

    return LoadedCatBoost43Models(
        models=MappingProxyType(loaded),
        feature_names=FORMAL_FEATURE_NAMES,
        label_order=FORMAL_LABEL_ORDER,
        manifest=MappingProxyType(dict(manifest)),
        model_directory=directory,
    )
