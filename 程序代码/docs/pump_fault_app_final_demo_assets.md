# 水泵故障诊断软件最终演示材料清单

## 1. 主演示样本

- 类别：`汽蚀`
- 来源：`Motor-4 / 70 / 出口汽蚀5`
- 文件路径：`/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀5/振动_电机4_70_时域-出口汽蚀5-通道4.csv`
- 原始采样率：`20000 Hz`
- 转速：`2070 rpm`
- 固定列名：`signal_column="0"`，`time_column="time"`
- 正式结果：`status=diagnosed`，预测类别 `汽蚀`，有效窗口数 `119`

## 2. 备用样本

- `Motor-4 / 70 / 出口汽蚀2`
- `Motor-4 / 70 / 转子不平衡2`
- `Motor-2 / 100 / 轴承故障2`

以 `configs/demo_samples.json` 为准。

## 3. 核心演示产物

- Word 报告：`demo_outputs/reports/primary_demo_report.docx`
- Word 报告 PDF 预览：`demo_outputs/reports/primary_demo_report.pdf`
- 截图目录：`demo_outputs/screenshots/`
- 自检原始结果：`demo_outputs/reports/self_check_result.json`
- demo package 检查原始结果：`demo_outputs/reports/demo_package_check.txt`

## 4. 已生成截图

- `01_home_page.png`
- `02_single_diagnosis_upload.png`
- `12_word_report_cover.png`
- `13_word_report_conclusion.png`
- `14_word_report_signal_figures.png`
- `15_self_check_result.png`
- `16_demo_package_check_result.png`

## 5. 需手动补截的 UI 结果图

- `03_single_diagnosis_result_cards.png`
- `04_class_probability_distribution.png`
- `05_window_prediction_distribution.png`
- `06_time_domain_waveform.png`
- `07_frequency_spectrum_0_5000hz.png`
- `08_envelope_spectrum.png`
- `09_wavelet_packet_energy.png`
- `10_report_view.png`
- `11_word_export_button.png`

原因：本轮未绕过 service 层或底层算法，UI 结果页截图需在已启动的 Streamlit 界面中完成一次真实单文件诊断后手动截取。

## 6. 启动与检查命令

```bash
cd /Users/hewenhao/Documents/特征提取/程序代码
./scripts/run_pump_fault_self_check.sh
./scripts/check_pump_fault_demo_package.sh configs/demo_package.json
./scripts/run_pump_fault_demo.sh
```

## 7. 演示注意事项

- 不要修改正式参数：`12000 Hz`、`10~5000 Hz`、`2400/1200`、`db6`、`2000~5000 Hz`
- 不要切换模型 bundle 或标签顺序
- 不要隐藏 `runtime_warnings`
- 对 `714` 条运行告警的解释应统一为：正式链路保留了运行时数值告警统计，结果已正常生成且可用于辅助诊断，但仍建议结合工况与现场记录复核
