#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/../.venv/bin/python"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/demo_package.json}"

if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "[failed] 未找到虚拟环境 Python：${VENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[failed] 未找到 demo package 配置：${CONFIG_PATH}" >&2
  echo "请先复制 configs/demo_package.example.json 为 configs/demo_package.json 并按本机环境修改。" >&2
  exit 1
fi

export PROJECT_ROOT
export CONFIG_PATH

"${VENV_PYTHON}" - <<'PY'
import json
import os
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
config_path = Path(os.environ["CONFIG_PATH"]).resolve()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def print_item(status: str, name: str, message: str) -> None:
    print(f"[{status}] {name}: {message}")


payload = json.loads(config_path.read_text(encoding="utf-8"))
failures = 0
warnings = 0

required_keys = [
    "demo_name",
    "description",
    "model_bundle_path",
    "demo_sample_config_path",
    "expected_ui_entry",
    "expected_cli_entry",
    "expected_output_dir",
    "expected_report_name",
    "required_files",
    "optional_files",
    "screenshot_output_dir",
]
for key in required_keys:
    if key not in payload:
        failures += 1
        print_item("failed", "config_key", f"缺少字段 {key}")

if failures:
    sys.exit(1)

print_item("passed", "demo_name", payload["demo_name"])
print_item("passed", "config_path", str(config_path))

for raw in payload["required_files"]:
    path = resolve_path(raw)
    if path.exists():
        print_item("passed", "required_file", str(path))
    else:
        failures += 1
        print_item("failed", "required_file", str(path))

for raw in payload.get("optional_files", []):
    path = resolve_path(raw)
    if path.exists():
        print_item("passed", "optional_file", str(path))
    else:
        warnings += 1
        print_item("warning", "optional_file", f"缺失可选材料 {path}")

model_bundle_path = resolve_path(payload["model_bundle_path"])
if model_bundle_path.exists():
    print_item("passed", "model_bundle_path", str(model_bundle_path))
else:
    failures += 1
    print_item("failed", "model_bundle_path", str(model_bundle_path))

demo_sample_config_path = resolve_path(payload["demo_sample_config_path"])
if demo_sample_config_path.exists():
    print_item("passed", "demo_sample_config_path", str(demo_sample_config_path))
    demo_sample_payload = json.loads(demo_sample_config_path.read_text(encoding="utf-8"))
    samples = demo_sample_payload.get("samples", [])
    if not samples:
        warnings += 1
        print_item("warning", "demo_sample_samples", "demo sample 配置中没有 samples")
    for sample in samples:
        file_path = resolve_path(sample["file_path"])
        if file_path.exists():
            print_item("passed", "demo_sample_file", str(file_path))
        else:
            failures += 1
            print_item("failed", "demo_sample_file", str(file_path))
else:
    failures += 1
    print_item("failed", "demo_sample_config_path", str(demo_sample_config_path))

screenshot_output_dir = resolve_path(payload["screenshot_output_dir"])
if screenshot_output_dir.exists():
    print_item("passed", "screenshot_output_dir", str(screenshot_output_dir))
else:
    warnings += 1
    print_item("warning", "screenshot_output_dir", f"截图输出目录不存在，可按需创建：{screenshot_output_dir}")

summary = f"failures={failures}, warnings={warnings}"
if failures:
    print_item("failed", "demo_package_summary", summary)
    sys.exit(1)

if warnings:
    print_item("warning", "demo_package_summary", summary)
else:
    print_item("passed", "demo_package_summary", summary)
PY
