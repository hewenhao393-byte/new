from __future__ import annotations

import importlib.util
import argparse
import json
import os
from pathlib import Path
from typing import Any

from pump_fault_app.domain.formal_contract import (
    FORMAL_FEATURE_NAMES,
    FORMAL_LABEL_ORDER,
    FORMAL_V3_CONTRACT,
    validate_feature_names,
    validate_label_order,
)
from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest
from pump_fault_app.prediction import DEFAULT_MODEL_DIRECTORY, load_catboost43_models
from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION

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
    model_directory: str | Path | None = None,
) -> dict[str, Any]:
    deployment_directory = Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    items: list[dict[str, str]] = []

    _record(items, "formal_contract", "passed", "formal inference contract loaded")
    _record(
        items,
        "version_information",
        "passed",
        (
            f"app={APP_VERSION}; model={MODEL_VERSION}; "
            f"features={FEATURE_VERSION}; contract={INFERENCE_CONTRACT_VERSION}"
        ),
    )
    try:
        FORMAL_V3_CONTRACT.validate()
        validate_feature_names(FORMAL_FEATURE_NAMES)
        validate_label_order(FORMAL_LABEL_ORDER)
        _record(items, "formal_parameters", "passed", "12000 Hz, 5-5000 Hz, 4800/2400, db6 level 3, 43 features")
    except Exception as exc:
        _record(items, "formal_parameters", "failed", str(exc))

    if deployment_directory.exists():
        _record(items, "deployment_models_exist", "passed", str(deployment_directory))
        try:
            loaded = load_catboost43_models(deployment_directory)
            _record(items, "deployment_model_contract", "passed", f"validated {', '.join(loaded.models)} hashes, features and classes")
        except Exception as exc:
            _record(items, "deployment_model_contract", "failed", str(exc))
    else:
        _record(items, "deployment_models_exist", "failed", f"missing deployment model directory: {deployment_directory}")
        _record(items, "deployment_model_contract", "failed", "model validation skipped because deployment directory is missing")

    for module_name in ("numpy", "pandas", "scipy", "catboost", "streamlit"):
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
            _run_demo_diagnosis_check(items, samples, deployment_directory)
        except Exception as exc:
            _record(items, "demo_sample_config", "failed", str(exc))
            _record(items, "demo_sample_minimal_diagnosis", "warning", "demo diagnosis skipped because demo config could not be loaded")

    overall_status = "passed"
    if any(item["status"] == "failed" for item in items):
        overall_status = "failed"
    elif any(item["status"] == "warning" for item in items):
        overall_status = "warning"

    return {"overall_status": overall_status, "items": items}


def _run_demo_diagnosis_check(items: list[dict[str, str]], samples: list[dict[str, Any]], model_directory: Path) -> None:
    if not samples:
        _record(items, "demo_sample_minimal_diagnosis", "warning", "demo sample list is empty")
        return
    sample = samples[0]
    channel_rows = sample.get("channels")
    if not isinstance(channel_rows, list) or not channel_rows:
        _record(items, "demo_sample_minimal_diagnosis", "warning", "demo sample does not define V3 channels")
        return
    channels = tuple(
        ChannelInput(
            str(row["channel"]).upper(),
            Path(str(row["file_path"])),
            row.get("signal_column"),
            row.get("time_column"),
        )
        for row in channel_rows
    )
    missing_paths = [str(item.file_path) for item in channels if not item.file_path.exists()]
    if missing_paths:
        _record(items, "demo_sample_minimal_diagnosis", "warning", f"demo sample files not found: {missing_paths}")
        return
    result = run_single_diagnosis(
        AppSingleRunRequest(
            MultiChannelInferenceRequest(
                channels,
                sampling_rate_hz=int(sample["sampling_rate"]),
                rpm=float(sample["rpm"]),
                model_directory=model_directory,
            )
        )
    )
    status = "passed" if result.inference_result.status == "diagnosed" else "warning"
    _record(items, "demo_sample_minimal_diagnosis", status, result.inference_result.status)


def _record_dependency(items: list[dict[str, str]], module_name: str, *, check_name: str | None = None) -> None:
    if importlib.util.find_spec(module_name):
        _record(items, check_name or f"dependency_{module_name}", "passed", f"{module_name} import available")
    else:
        _record(items, check_name or f"dependency_{module_name}", "failed", f"{module_name} import missing")


def _record(items: list[dict[str, str]], check_name: str, status: str, message: str) -> None:
    items.append({"check_name": check_name, "status": status, "message": message})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CatBoost43 V3 system self-check")
    parser.add_argument("--demo-config")
    parser.add_argument("--model-directory")
    args = parser.parse_args(argv)
    payload = run_system_self_check(
        demo_config_path=args.demo_config,
        model_directory=args.model_directory,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["overall_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
