# 水泵故障诊断软件最终截图清单

## 1. 截图前固定条件

- 先运行 `./scripts/run_pump_fault_self_check.sh`
- 再运行 `./scripts/check_pump_fault_demo_package.sh configs/demo_package.json`
- 最后运行 `./scripts/run_pump_fault_demo.sh`
- 主演示样本固定为 `Motor-4 / 70 / 汽蚀 / 出口汽蚀5`
- 原始文件路径：`/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀5/振动_电机4_70_时域-出口汽蚀5-通道4.csv`
- 主演示输入参数：原始采样率 `20000 Hz`，转速 `2070 rpm`，`signal_column="0"`，`time_column="time"`
- 正式推理参数保持：`12000 Hz`、`10~5000 Hz`、`2400/1200`、`db6`、`2000~5000 Hz`

## 2. 最终截图文件名与状态

| 文件名 | 内容 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| `01_home_page.png` | 系统首页 | 已生成 | 可直接用于论文和答辩 |
| `02_single_diagnosis_upload.png` | 单文件诊断输入页 | 已生成 | 展示上传区与参数输入区 |
| `03_single_diagnosis_result_cards.png` | 结果核心指标卡 | 需手动截图 | 需在 UI 中完成一次正式诊断后截取 |
| `04_class_probability_distribution.png` | 六分类概率分布图 | 需手动截图 | 需在单文件诊断结果页截取 |
| `05_window_prediction_distribution.png` | 窗口级预测类别分布 | 需手动截图 | 需在单文件诊断结果页截取 |
| `06_time_domain_waveform.png` | 时域波形 | 需手动截图 | 需在单文件诊断结果页截取 |
| `07_frequency_spectrum_0_5000hz.png` | `0~5000 Hz` 频谱图 | 需手动截图 | 需在单文件诊断结果页截取 |
| `08_envelope_spectrum.png` | 包络谱图 | 需手动截图 | 需在详细分析折叠区截取 |
| `09_wavelet_packet_energy.png` | 小波包能量占比图 | 需手动截图 | 需在详细分析折叠区截取 |
| `10_report_view.png` | 报告页总览 | 当前为占位截图 | 已生成的是“暂无诊断结果”空页面，最终答辩版需手动补截有结果页面 |
| `11_word_export_button.png` | Word 报告导出按钮 | 需手动截图 | 需在报告页有结果时截取导出区域 |
| `12_word_report_cover.png` | Word 报告首页 | 已生成 | 来自正式导出的 `primary_demo_report.docx` |
| `13_word_report_conclusion.png` | Word 报告诊断结论 | 已生成 | 展示基本信息与诊断结论 |
| `14_word_report_signal_figures.png` | Word 报告信号分析图部分 | 已生成 | 已整理时域、频谱、包络谱、小波包图 |
| `15_self_check_result.png` | 系统自检结果 | 已生成 | 对应 `run_pump_fault_self_check.sh` 本轮输出 |
| `16_demo_package_check_result.png` | demo package 检查结果 | 已生成 | 对应 `check_pump_fault_demo_package.sh` 本轮输出 |

## 3. 每张图的使用建议

### `01_home_page.png`

- 截图位置：Streamlit 首页
- 截图目的：展示软件名称、六类故障、处理流程和正式参数
- 论文可放章节：第5章 软件设计与实现
- 答辩讲法：先说明这是正式软件首页，再说明系统处理流程和统一参数契约

### `02_single_diagnosis_upload.png`

- 截图位置：左侧导航“单文件诊断”
- 截图目的：展示文件上传、采样率、转速输入
- 论文可放章节：第5章 输入与交互设计
- 答辩讲法：强调普通用户只需要提供振动文件、原始采样率和转速

### `03_single_diagnosis_result_cards.png`

- 截图位置：单文件诊断结果顶部四张指标卡
- 截图目的：突出 `汽蚀`、高置信度、窗口一致率和有效窗口数
- 论文可放章节：第5章 结果展示模块
- 答辩讲法：先讲最终类别，再讲可靠性信息

### `04_class_probability_distribution.png`

- 截图位置：单文件诊断结果页概率图
- 截图目的：展示正式六分类概率输出
- 论文可放章节：第5章 诊断结果可视化
- 答辩讲法：说明系统不是只给单一标签，而是输出完整概率分布

### `05_window_prediction_distribution.png`

- 截图位置：窗口级预测分布图
- 截图目的：说明最终结果来自多窗口融合
- 论文可放章节：第5章 多窗口结果组织
- 答辩讲法：解释为什么不是单窗口直接判断

### `06_time_domain_waveform.png`

- 截图位置：主结果区时域图
- 截图目的：展示 visualization 契约中的时域显示数据
- 论文可放章节：第5章 信号分析展示
- 答辩讲法：说明时域图用于辅助复核，不单独决定故障类别

### `07_frequency_spectrum_0_5000hz.png`

- 截图位置：主结果区频谱图
- 截图目的：展示正式分析频带内的频谱信息
- 论文可放章节：第5章 频域分析展示
- 答辩讲法：联系 `10~5000 Hz` 正式分析带宽说明

### `08_envelope_spectrum.png`

- 截图位置：详细分析折叠区
- 截图目的：展示包络谱辅助信息
- 论文可放章节：第5章 包络分析展示
- 答辩讲法：说明该图来自正式 visualization 数据，不是 UI 现场重新计算

### `09_wavelet_packet_energy.png`

- 截图位置：详细分析折叠区
- 截图目的：展示 `db6` 三层小波包能量占比
- 论文可放章节：第5章 多域特征展示
- 答辩讲法：可结合 21 维特征中的小波包能量比解释

### `10_report_view.png`

- 截图位置：左侧导航“诊断报告”
- 截图目的：展示结果复核与导出页面
- 论文可放章节：第5章 报告模块
- 答辩讲法：强调报告页不重新诊断，只展示正式结果

### `11_word_export_button.png`

- 截图位置：报告页导出区域
- 截图目的：展示单文件 Word 报告导出入口
- 论文可放章节：第5章 报告导出实现
- 答辩讲法：说明 UI 与 CLI 共用统一报告导出服务

### `12_word_report_cover.png`

- 截图位置：导出的 `primary_demo_report.docx` 首页
- 截图目的：展示报告标题、基本信息与正式输出样式
- 论文可放章节：第5章 报告格式设计
- 答辩讲法：说明报告来自同一套正式视图数据

### `13_word_report_conclusion.png`

- 截图位置：Word 报告“诊断结论”部分
- 截图目的：展示最终类别、置信度和建议
- 论文可放章节：第5章 结果输出设计
- 答辩讲法：说明软件输出不只是一行标签，还包括结构化诊断摘要

### `14_word_report_signal_figures.png`

- 截图位置：Word 报告图形章节
- 截图目的：展示时域、频谱、包络谱、小波包图已进入正式报告
- 论文可放章节：第5章 报告导出示例
- 答辩讲法：说明图形与 UI 同源，均来自正式 visualization 契约

### `15_self_check_result.png`

- 截图位置：系统自检输出
- 截图目的：展示部署完整性检查
- 论文可放章节：第5章 系统验收与可靠性
- 答辩讲法：答辩前可快速确认模型、依赖、演示样本配置是否齐全

### `16_demo_package_check_result.png`

- 截图位置：demo package 检查输出
- 截图目的：展示演示材料完整性检查
- 论文可放章节：第5章 演示版本固化
- 答辩讲法：说明答辩材料、脚本、样本和输出目录都已固化

## 4. 手动补截建议顺序

1. 启动 UI，进入“单文件诊断”
2. 上传主演示样本 `出口汽蚀5`
3. 输入 `20000 Hz` 和 `2070 rpm`
4. 依次补截 `03` 到 `11`
5. 最后确认 `10_report_view.png` 为“有结果”的正式页面，而不是空页面

## 5. 当前自动生成产物位置

- 截图目录：`demo_outputs/screenshots/`
- Word 报告：`demo_outputs/reports/primary_demo_report.docx`
- 自检结果原始文件：`demo_outputs/reports/self_check_result.json`
- demo package 检查原始文件：`demo_outputs/reports/demo_package_check.txt`
