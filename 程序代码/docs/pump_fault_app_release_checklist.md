# 水泵故障诊断软件发布清单

## 1. 代码状态检查

- 当前工作区无未确认风险修改
- 主流程相关测试已通过
- 旧实验脚本未被误改为软件入口

## 2. 正式参数检查

正式推理必须保持：

- `12000 Hz`
- `10~5000 Hz`
- `2400/1200`
- `db6`
- `2000~5000 Hz`

## 3. 模型 bundle 检查

- `bp_bundle.joblib` 路径存在
- bundle 字段完整
- 21维特征顺序一致
- 六类标签顺序一致

## 4. 依赖环境检查

- `numpy`
- `pandas`
- `scipy`
- `joblib`
- `streamlit`
- `python-docx`
- `matplotlib`

## 5. 演示样本配置检查

- `configs/demo_samples.json` 已存在
- `file_path` 指向本机真实样本
- `sampling_rate`、`rpm`、`signal_column` 已核对

## 6. 单文件诊断检查

- UI 单文件诊断能运行
- CLI 单文件诊断能运行
- 概率分布与图形能显示

## 7. Word 报告导出检查

- 报告页 Word 导出可用
- CLI `--output-report` 可用
- `.docx` 文件非空

## 8. UI 启动检查

- `streamlit_app.py` 可启动
- 首页、单文件诊断页、报告页可访问

## 9. CLI 诊断检查

- `pump_fault_app.app.cli` 可调用
- `--self-check` 可调用
- `--output-report` 可调用

## 10. 系统自检检查

运行：

```bash
../.venv/bin/python -m pump_fault_app.app.cli --self-check
```

或：

```bash
./scripts/run_pump_fault_self_check.sh
```

## 11. 测试命令

```bash
../.venv/bin/python -m pytest tests/test_pump_fault_app_signal_reader.py tests/test_pump_fault_app_signal_quality.py tests/test_pump_fault_app_preprocessing.py tests/test_pump_fault_app_windowing.py tests/test_pump_fault_app_feature_extraction.py tests/test_pump_fault_app_prediction.py tests/test_pump_fault_app_fusion.py tests/test_pump_fault_app_inference.py tests/test_pump_fault_app_reporting.py tests/test_pump_fault_app_cli.py tests/test_pump_fault_app_batch.py tests/test_pump_fault_app_batch_manifest.py tests/test_pump_fault_app_batch_cli.py tests/test_pump_fault_app_export.py tests/test_pump_fault_app_services.py tests/test_pump_fault_app_ui.py tests/test_pump_fault_app_config.py tests/test_pump_fault_app_logging.py tests/test_pump_fault_app_records.py tests/test_inference_contract.py tests/test_pump_fault_app_self_check.py tests/test_pump_fault_app_end_to_end.py -q
```

## 12. 预期测试结果

- 最近一次完整验收：`135 passed`

## 13. 常见问题

- 模型 bundle 缺失：检查 V2 实验目录
- demo sample 配置缺失：复制示例 JSON 并修改路径
- 自检 warning：优先检查演示样本和本机路径
- Word 导出失败：检查 `python-docx` 和 `matplotlib`

## 14. 发布前不允许改动的内容

- BP 模型
- 21维特征顺序
- 预处理参数
- 窗口参数
- 标签顺序
