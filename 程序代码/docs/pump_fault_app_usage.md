# 水泵故障诊断软件使用说明

## 1. 软件简介

`pump_fault_app` 用于对输入振动信号进行单文件六分类故障诊断，并支持：

- 单文件正式推理
- Streamlit 界面展示
- CLI 单文件诊断
- Word 诊断报告导出
- 系统自检

## 2. 支持的六类故障

- 正常
- 转子不平衡
- 联轴器不对中
- 松动
- 轴承故障
- 汽蚀

## 3. 正式推理参数

- 目标采样率：`12000 Hz`
- 带通范围：`10~5000 Hz`
- 窗口长度：`2400`
- 滑动步长：`1200`
- 包络频带：`2000~5000 Hz`
- 小波包：`db6` 三层

旧 `4096/2048`、`db4` 和旧标签顺序只用于实验追溯，不得作为正式软件推理入口。

## 4. 输入数据格式要求

支持：

- 单列 CSV
- 带时间列的 CSV
- 多通道 CSV
- TXT 数值文件

建议至少提供：

- 振动信号列
- 原始采样率
- 转速 `rpm`

## 5. 为什么必须填写转速 rpm

当前正式 21 维特征中包含转频相关指标，例如：

- `rot_2x_1x_ratio`
- `rot_3x_1x_ratio`
- `harmonic_energy_ratio_1x_5x`

因此 `rpm` 是正式推理链路的必要输入。缺少 `rpm` 会导致特征口径与模型训练口径不一致。

## 6. Streamlit 单文件诊断流程

在 `程序代码/` 目录运行：

```bash
../.venv/bin/python -m streamlit run pump_fault_app/ui/streamlit_app.py
```

单文件诊断步骤：

1. 上传振动文件
2. 输入采样率和转速
3. 点击“开始诊断”
4. 查看诊断结果、概率分布和信号分析图
5. 在报告页导出 Word 报告

## 7. CLI 单文件诊断

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1450
```

## 8. CLI 导出 Word 报告

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1450 \
  --output-report ./single_report.docx
```

## 9. 系统自检命令

```bash
../.venv/bin/python -m pump_fault_app.app.cli --self-check
```

如果要显式指定演示样本配置或模型路径：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --self-check \
  --demo-config ./configs/demo_samples.json \
  --model-bundle ../实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib
```

## 10. 演示样本配置

演示样本建议通过 JSON 文件指定，示例文件：

- `configs/demo_samples.example.json`

推荐做法：

1. 复制为 `configs/demo_samples.json`
2. 将 `file_path` 改为本机实际样本路径
3. 或设置环境变量 `PUMP_FAULT_APP_DEMO_CONFIG`

用户替换为自己的振动数据时，应同步更新：

- `file_path`
- `sampling_rate`
- `rpm`
- `signal_column`

## 11. Word 报告导出说明

Word 报告会基于正式单文件诊断结果生成，包含：

- 基本信息
- 诊断结论
- 六类概率分布
- 窗口预测分布
- 信号分析图
- 说明与适用范围

Word 报告不会重新做诊断，也不会重新读取原始信号。

## 12. 常见错误与解决方法

- 模型 bundle 缺失：检查 `bp_bundle.joblib` 路径是否存在。
- 演示样本配置缺失：复制 `configs/demo_samples.example.json` 为实际配置并修改路径。
- 采样率或转速未填写：正式推理无法构造完整特征。
- 信号质量拒绝：说明样本长度、幅值或有效性不满足正式诊断条件。
- Word 导出失败：检查 `python-docx` 和 `matplotlib` 是否已安装。

## 13. 当前适用范围与局限性

当前模型主要适用于与训练数据采集条件相近的水泵振动信号。对于未知泵型、未知测点、不同传感器或明显不同工况，诊断结果应作为辅助参考，建议结合现场检查复核。
