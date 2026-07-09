# 水泵故障诊断软件答辩演示指南

## 1. 演示前准备

- 确认虚拟环境存在
- 确认模型 bundle 存在
- 准备好演示样本配置
- 先运行一次系统自检

## 2. 如何准备 demo sample

1. 复制 `configs/demo_samples.example.json`
2. 重命名为 `configs/demo_samples.json`
3. 将 `file_path` 改为本机实际样本路径
4. 核对 `sampling_rate`、`rpm`、`signal_column`

如果希望一次生成本地演示配置模板，可先运行：

```bash
./scripts/init_pump_fault_demo_package.sh
```

当前主演示样本已固定为：

- `Motor-4 / 70 / 汽蚀`
- `出口汽蚀5`
- 原始采样率 `20000 Hz`
- 转速 `2070 rpm`
- `signal_column="0"`，`time_column="time"`
- 正式单文件诊断结果：`汽蚀`
- 综合置信度接近 `1.0`
- 概率差接近 `1.0`
- 窗口一致率 `1.0000`
- 有效窗口数 `119`

备用样本见：

- `configs/demo_samples.json`
- `docs/pump_fault_app_demo_sample_selection.md`

## 3. 如何运行 self-check

```bash
./scripts/run_pump_fault_self_check.sh
```

或者：

```bash
../.venv/bin/python -m pump_fault_app.app.cli --self-check
```

## 4. 如何启动 UI

```bash
./scripts/run_pump_fault_demo.sh
```

脚本会：

- 如存在 `configs/demo_package.json`，先执行演示数据包检查
- 检查虚拟环境
- 检查模型 bundle
- 检查 demo sample 配置
- 先运行 self-check
- 启动 Streamlit

## 5. 如何上传振动信号

在“单文件诊断”页面上传：

- CSV
- TXT
- WAV（如当前环境支持）

## 6. 如何输入采样率和转速

- 采样率填写原始采样率
- 转速填写当前记录对应的真实转速 `rpm`

## 7. 如何查看诊断结果

重点展示：

- 预测故障类别
- 综合置信度
- 有效窗口数
- 信号质量

当前主演示样本推荐讲法：

- 期望类别和预测类别均为 `汽蚀`
- 六分类融合结果高度集中在 `汽蚀`
- 全部 `119` 个有效窗口均支持同一结论

## 8. 如何查看时域、频谱、包络谱、小波包图

建议展示顺序：

- 时域波形
- `0~5000 Hz` 频谱
- 包络谱
- 小波包能量占比

## 9. 如何导出 Word 报告

在报告页点击“导出 Word 报告”即可。

也可以使用 CLI：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1450 \
  --output-report ./single_report.docx
```

## 10. 建议展示页面顺序

1. 首页
2. 单文件诊断页
3. 上传振动信号
4. 输入采样率和转速
5. 查看六分类结果
6. 查看概率分布
7. 查看时域和频谱
8. 查看包络谱和小波包能量
9. 打开报告页
10. 导出 Word 报告
11. 展示系统自检结果

## 11. 可能被问到的问题和回答提示

### 为什么会出现 `714` 条运行告警

建议答辩话术：

- 这些告警是正式推理过程中记录的运行时数值告警统计，不是界面额外生成的提示
- 告警并没有阻断正式诊断，本次结果状态仍为 `diagnosed`
- 该样本的最终类别、融合概率、窗口一致率和 Word 报告都已正常生成，说明当前正式链路仍然完成了有效输出
- 这类告警应解释为“软件保留底层运行痕迹，结果可用，但仍建议结合工况与现场记录复核”，而不是简单说成“完全没有问题”

### 为什么必须填写 rpm

因为正式特征中包含转频相关指标，`rpm` 是正式推理链路必要输入。

### 为什么只允许这一套参数

因为软件推理必须与 V2 正式训练口径一致，不能混用旧 `4096/2048`、`db4` 或旧标签顺序。

### 结果是否能替代现场检测

不能。当前结果用于辅助状态判断和检修排查，最终仍需结合现场检查。

### 适用于哪些工况

当前模型主要适用于与训练数据采集条件相近的水泵振动信号。对于未知泵型、未知测点、不同传感器或明显不同工况，应谨慎使用。

## 12. 正式参数提醒

- `12000 Hz`
- `10~5000 Hz`
- `2400/1200`
- `db6`
- `2000~5000 Hz`

## 13. 最终截图材料位置

- 截图目录：`demo_outputs/screenshots/`
- Word 报告：`demo_outputs/reports/primary_demo_report.docx`
- 截图清单：`docs/pump_fault_app_screenshot_checklist.md`
