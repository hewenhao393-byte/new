from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from pump_diagnosis.inference_contract import (
    FORMAL_LABEL_ORDER,
    FORMAL_MODEL_BUNDLE_PATH,
    FORMAL_V2_CONTRACT,
    validate_feature_names,
    validate_label_order,
    validate_model_bundle,
)
from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis

DEFAULT_DEMO_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "demo_samples.json"
DEMO_CONFIG_ENV_VAR = "PUMP_FAULT_APP_DEMO_CONFIG"


def resolve_demo_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get(DEMO_CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_DEMO_CONFIG_PATH


def load_demo_sample_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_demo_config_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("demo sample config must be a JSON object containing a samples list")
    return payload


def run_system_self_check(
    *,
    demo_config_path: str | Path | None = None,
    model_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle_path = Path(model_bundle_path) if model_bundle_path is not None else FORMAL_MODEL_BUNDLE_PATH
    items: list[dict[str, str]] = []

    _record(items, "formal_contract", "passed", "formal inference contract loaded")
    try:
        FORMAL_V2_CONTRACT.validate()
        validate_feature_names(FORMAL_V2_CONTRACT.feature_names)
        validate_label_order(FORMAL_LABEL_ORDER)
        _record(items, "formal_parameters", "passed", "12000 Hz, 10-5000 Hz, 2400/1200, db6, 2000-5000 Hz")
    except Exception as exc:
        _record(items, "formal_parameters", "failed", str(exc))

    if bundle_path.exists():
        _record(items, "model_bundle_exists", "passed", str(bundle_path))
        try:
            validate_model_bundle(bundle_path)
            _record(items, "model_bundle_contract", "passed", "bundle keys, labels, features and predict_proba validated")
        except Exception as exc:
            _record(items, "model_bundle_contract", "failed", str(exc))
    else:
        _record(items, "model_bundle_exists", "failed", f"missing model bundle: {bundle_path}")
        _record(items, "model_bundle_contract", "failed", "bundle validation skipped because model bundle is missing")

    for module_name in ("numpy", "pandas", "scipy", "joblib", "streamlit"):
        _record_dependency(items, module_name)
    for module_name in ("docx", "matplotlib"):
        _record_dependency(items, module_name, check_name=f"word_dependency_{module_name}")

    try:
        from pump_fault_app.app.cli import build_parser

        build_parser()
        _record(items, "cli_entrypoint", "passed", "CLI parser is callable")
    except Exception as exc:
        _record(items, "cli_entrypoint", "failed", str(exc))

    config_path = resolve_demo_config_path(demo_config_path)
    if not config_path.exists():
        _record(items, "demo_sample_config", "warning", f"demo sample config not found: {config_path}")
    else:
        try:
            payload = load_demo_sample_config(config_path)
            samples = payload["samples"]
            _record(items, "demo_sample_config", "passed", f"loaded {len(samples)} demo sample entries from {config_path}")
            _run_demo_diagnosis_check(items, samples, bundle_path)
        except Exception as exc:
            _record(items, "demo_sample_config", "failed", str(exc))
            _record(items, "demo_sample_minimal_diagnosis", "warning", "demo diagnosis skipped because demo config could not be loaded")

    overall_status = "passed"
    if any(item["status"] == "failed" for item in items):
        overall_status = "failed"
    elif any(item["status"] == "warning" for item in items):
        overall_status = "warning"

    return {"overall_status": overall_status, "items": items}


def _run_demo_diagnosis_check(items: list[dict[str, str]], samples: list[dict[str, Any]], bundle_path: Path) -> None:
    if not samples:
        _record(items, "demo_sample_minimal_diagnosis", "warning", "demo sample list is empty")
        return
    sample = samples[0]
    sample_path = Path(str(sample["file_path"]))
    if not sample_path.exists():
        _record(items, "demo_sample_minimal_diagnosis", "warning", f"demo sample file not found: {sample_path}")
        return
    if not bundle_path.exists():
        _record(items, "demo_sample_minimal_diagnosis", "warning", "demo diagnosis skipped because model bundle is missing")
        return
    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=sample_path,
            sampling_rate_hz=int(sample["sampling_rate"]),
            rpm=float(sample["rpm"]),
            signal_column=sample.get("signal_column"),
            time_column=sample.get("time_column"),
            device_id=sample.get("device_id"),
            measurement_position=sample.get("measurement_position"),
            model_bundle_path=bundle_path,
        )
    )
    status = "passed" if result.summary.success else "warning"
    _record(items, "demo_sample_minimal_diagnosis", status, result.summary.message)


def _record_dependency(items: list[dict[str, str]], module_name: str, *, check_name: str | None = None) -> None:
    if importlib.util.find_spec(module_name):
        _record(items, check_name or f"dependency_{module_name}", "passed", f"{module_name} import available")
    else:
        _record(items, check_name or f"dependency_{module_name}", "failed", f"{module_name} import missing")


def _record(items: list[dict[str, str]], check_name: str, status: str, message: str) -> None:
    items.append({"check_name": check_name, "status": status, "message": message})
