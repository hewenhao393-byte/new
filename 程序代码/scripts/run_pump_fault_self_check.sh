#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/../.venv/bin/python"

MODEL_BUNDLE="${1:-}"
DEMO_CONFIG="${2:-}"

if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "未找到虚拟环境 Python：${VENV_PYTHON}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

CMD=("${VENV_PYTHON}" -m pump_fault_app.app.cli --self-check)
if [[ -n "${MODEL_BUNDLE}" ]]; then
  CMD+=(--model-bundle "${MODEL_BUNDLE}")
fi
if [[ -n "${DEMO_CONFIG}" ]]; then
  CMD+=(--demo-config "${DEMO_CONFIG}")
fi

echo "运行系统自检..."
echo "命令：${CMD[*]}"
"${CMD[@]}"
