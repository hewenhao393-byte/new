#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/../.venv/bin/python"
MODEL_BUNDLE_DEFAULT="${PROJECT_ROOT}/../实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib"
DEMO_CONFIG_DEFAULT="${PROJECT_ROOT}/configs/demo_samples.json"
DEMO_PACKAGE_CONFIG_DEFAULT="${PROJECT_ROOT}/configs/demo_package.json"

MODEL_BUNDLE="${PUMP_FAULT_APP_MODEL_BUNDLE:-${MODEL_BUNDLE_DEFAULT}}"
DEMO_CONFIG="${PUMP_FAULT_APP_DEMO_CONFIG:-${DEMO_CONFIG_DEFAULT}}"
DEMO_PACKAGE_CONFIG="${PUMP_FAULT_APP_DEMO_PACKAGE_CONFIG:-${DEMO_PACKAGE_CONFIG_DEFAULT}}"

if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "未找到虚拟环境 Python：${VENV_PYTHON}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "项目目录：${PROJECT_ROOT}"
echo "Python：${VENV_PYTHON}"

if [[ ! -f "${MODEL_BUNDLE}" ]]; then
  echo "未找到正式模型 bundle：${MODEL_BUNDLE}" >&2
  exit 1
fi

if [[ ! -f "${DEMO_CONFIG}" ]]; then
  echo "未找到演示样本配置：${DEMO_CONFIG}" >&2
  echo "请先复制 configs/demo_samples.example.json 为 configs/demo_samples.json 并修改 file_path。" >&2
  exit 1
fi

if [[ -f "${DEMO_PACKAGE_CONFIG}" ]]; then
  echo "先执行演示数据包检查..."
  "${PROJECT_ROOT}/scripts/check_pump_fault_demo_package.sh" "${DEMO_PACKAGE_CONFIG}"
else
  echo "未找到 demo package 配置：${DEMO_PACKAGE_CONFIG}" >&2
  echo "可先运行 ./scripts/init_pump_fault_demo_package.sh 生成本地配置模板。" >&2
fi

echo "先执行系统自检..."
"${VENV_PYTHON}" -m pump_fault_app.app.cli \
  --self-check \
  --model-bundle "${MODEL_BUNDLE}" \
  --demo-config "${DEMO_CONFIG}"

echo
echo "自检完成，准备启动 Streamlit。"
echo "访问地址通常为：http://localhost:8501"
echo

exec "${VENV_PYTHON}" -m streamlit run pump_fault_app/ui/streamlit_app.py
