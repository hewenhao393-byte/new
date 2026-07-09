#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_SAMPLE_EXAMPLE="${PROJECT_ROOT}/configs/demo_samples.example.json"
DEMO_SAMPLE_TARGET="${PROJECT_ROOT}/configs/demo_samples.json"
DEMO_PACKAGE_EXAMPLE="${PROJECT_ROOT}/configs/demo_package.example.json"
DEMO_PACKAGE_TARGET="${PROJECT_ROOT}/configs/demo_package.json"

copy_if_missing() {
  local source_file="$1"
  local target_file="$2"

  if [[ -f "${target_file}" ]]; then
    echo "已存在，跳过：${target_file}"
    return 0
  fi

  cp "${source_file}" "${target_file}"
  echo "已生成：${target_file}"
}

copy_if_missing "${DEMO_SAMPLE_EXAMPLE}" "${DEMO_SAMPLE_TARGET}"
copy_if_missing "${DEMO_PACKAGE_EXAMPLE}" "${DEMO_PACKAGE_TARGET}"

echo
echo "请继续检查以下字段并改为本机真实路径："
echo "- configs/demo_samples.json 中的 file_path"
echo "- configs/demo_package.json 中的 model_bundle_path"
echo "- configs/demo_package.json 中的 demo_sample_config_path"
echo "- configs/demo_package.json 中的 expected_output_dir / screenshot_output_dir"
