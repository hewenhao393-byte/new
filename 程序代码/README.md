# 通道 4 水泵六分类故障诊断

在本文件所在的 `程序代码/` 目录运行以下命令。

如果你想先看“这套代码每个模块是干什么的”，先看这份总说明：

- [docs/pump_fault_app_walkthrough.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_walkthrough.md:1)
- [docs/pump_fault_app_usage.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_usage.md:1)
- [docs/pump_fault_app_acceptance.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_acceptance.md:1)
- [docs/pump_fault_app_demo_guide.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_demo_guide.md:1)
- [docs/pump_fault_app_release_checklist.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_release_checklist.md:1)
- [docs/pump_fault_app_screenshot_checklist.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_screenshot_checklist.md:1)
- [docs/pump_fault_app_thesis_figures.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_thesis_figures.md:1)
- [docs/pump_fault_app_defense_checklist.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_defense_checklist.md:1)
- [docs/pump_fault_app_demo_sample_selection.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_demo_sample_selection.md:1)
- [docs/pump_fault_app_final_demo_assets.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_final_demo_assets.md:1)
- [docs/pump_fault_app_architecture.md](docs/pump_fault_app_architecture.md)

## 一键演示

在 `程序代码/` 目录执行：

```bash
./scripts/run_pump_fault_demo.sh
```

脚本会：

- 检查虚拟环境
- 检查模型 bundle
- 检查 demo sample 配置
- 先执行 `self-check`
- 启动现有 Streamlit 入口 `pump_fault_app/ui/streamlit_app.py`

## 演示数据包配置

演示交付建议再维护一份独立的 demo package 配置，示例文件：

- `configs/demo_package.example.json`

建议复制为：

- `configs/demo_package.json`

该配置用于统一检查：

- 正式模型 bundle
- demo sample 配置
- 演示输出目录
- 截图输出目录
- README、使用说明、验收说明、演示指南和脚本等交付材料

如果本地还没有实际配置，可先运行：

```bash
./scripts/init_pump_fault_demo_package.sh
```

它会自动生成：

- `configs/demo_samples.json`
- `configs/demo_package.json`

## 演示数据包检查

在 `程序代码/` 目录执行：

```bash
./scripts/check_pump_fault_demo_package.sh configs/demo_package.json
```

脚本会检查：

- `required_files` 是否齐全
- `optional_files` 是否存在
- `demo_sample_config_path` 是否存在
- demo sample 中 `file_path` 指向的振动文件是否存在
- 截图输出目录是否已准备

## 演示前推荐执行顺序

```bash
./scripts/init_pump_fault_demo_package.sh
./scripts/run_pump_fault_self_check.sh
./scripts/check_pump_fault_demo_package.sh configs/demo_package.json
./scripts/run_pump_fault_demo.sh
```

## 当前主演示样本

当前固定主演示样本为：

- `Motor-4 / 70 / 汽蚀 / 出口汽蚀5`
- 原始文件：`/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀5/振动_电机4_70_时域-出口汽蚀5-通道4.csv`
- 原始采样率：`20000 Hz`
- 转速：`2070 rpm`
- 固定读取：`signal_column="0"`，`time_column="time"`

备用样本和筛选依据见：

- `configs/demo_samples.json`
- `docs/pump_fault_app_demo_sample_selection.md`

当前主演示样本正式结果：

- 期望类别：`汽蚀`
- 预测类别：`汽蚀`
- 综合置信度：接近 `1.0`
- 概率差：接近 `1.0`
- 窗口一致率：`1.0000`
- 有效窗口数：`119`
- visualization：四类图完整
- Word 报告：可正常导出

## 最终演示材料

本轮已经固化的正式演示材料见：

- `demo_outputs/screenshots/`
- `demo_outputs/reports/primary_demo_report.docx`
- `docs/pump_fault_app_screenshot_checklist.md`
- `docs/pump_fault_app_final_demo_assets.md`

其中：

- 首页、单文件输入页、Word 报告截图、自检截图和 demo package 检查截图已生成
- 单文件结果页和报告页有结果状态截图，需要在已启动的 Streamlit 界面中手动补截

关于 `runtime_warnings=714` 的答辩话术建议：

- 这是正式推理过程记录的运行时数值告警统计，不是界面额外生成的提示
- 本次正式结果状态仍为 `diagnosed`
- 预测类别、融合概率、窗口一致率、可视化和 Word 报告都已正常生成
- 因此可解释为“结果可用，但建议结合工况与现场记录复核”

## 软件骨架

新建的 `pump_fault_app/` 用于后续六分类故障诊断软件开发，目前只包含：

- 统一配置模块
- 六类标签和 21 维最终特征定义
- 数据结构
- 日志初始化
- 原始信号读取、质检、预处理、分窗、21维正式特征提取
- 正式 BP bundle 单窗预测、多窗融合、单条记录推理入口
- 单文件 CLI、批量 CLI
- Streamlit 界面骨架

### 运行骨架

在 `程序代码/` 目录执行：

`../.venv/bin/python -m pump_fault_app.app.bootstrap`

该命令会输出配置摘要，并在 `runtime_logs/` 下写入日志文件。

### 测试骨架

在 `程序代码/` 目录执行：

`../.venv/bin/python -m pytest tests/test_pump_fault_app_config.py tests/test_pump_fault_app_logging.py tests/test_pump_fault_app_records.py -q`

## Streamlit 启动

先安装依赖：

```bash
../.venv/bin/pip install -r requirements.txt
```

启动首页：

```bash
../.venv/bin/python -m streamlit run pump_fault_app/ui/streamlit_app.py
```

当前界面文件：

- `pump_fault_app/ui/streamlit_app.py`
- `pump_fault_app/ui/pages/single_diagnosis.py`
- `pump_fault_app/ui/pages/batch_diagnosis.py`
- `pump_fault_app/ui/pages/report_view.py`

界面层严格只调用 `pump_fault_app.services` 中的统一 service API。

## 正式推理契约

软件正式推理只允许使用 `pump_fault_app/domain/formal_contract.py` 中冻结的 V2 契约：

- 目标采样率：`12000 Hz`
- 带通范围：`10~5000 Hz`
- 窗口长度：`2400`
- 滑动步长：`1200`
- 包络频带：`2000~5000 Hz`
- 小波包：`db6` 三层
- 六类标签顺序：
  - `正常`
  - `转子不平衡`
  - `联轴器不对中`
  - `松动`
  - `轴承故障`
  - `汽蚀`

旧实验脚本中的其他窗口参数、标签顺序和小波参数不能作为软件正式推理依据。
旧训练脚本和探索脚本只作为实验追溯资料，不作为软件入口。

软件、模型、特征和推理契约版本统一定义在 `pump_fault_app/version.py`；
`pump_diagnosis/inference_contract.py` 仅保留为旧实验脚本的兼容导入入口。

## Python 入口

单条记录正式推理入口：

```python
from pathlib import Path

from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference
from pump_fault_app.reporting import build_diagnosis_summary

result = run_formal_inference(
    FormalInferenceRequest(
        file_path=Path("your_signal.csv"),
        sampling_rate_hz=12000,
        rpm=1500.0,
    )
)
summary = build_diagnosis_summary(result)
print(summary.as_dict())
```

批量推理入口：

```python
from pathlib import Path

from pump_fault_app.batch import BatchInferenceRequest, run_batch_inference

batch_result = run_batch_inference(
    BatchInferenceRequest(
        file_paths=(Path("a.csv"), Path("b.csv")),
        sampling_rate_hz=12000,
        rpm=1500.0,
    )
)
print(batch_result)
```

## 单文件命令行运行

在 `程序代码/` 目录执行：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1500
```

如果需要显式指定信号列、时间列或模型文件，可附加：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1500 \
  --signal-column 通道4 \
  --time-column time \
  --model-bundle ../实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib
```

如果需要同时导出单文件结果，可附加：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1500 \
  --output-dir ./single_outputs
```

`--output-dir` 表示导出根目录。每次运行会自动新建一个时间戳子目录，例如：

- `single_outputs/single_run_20260707_160000/`

该运行目录中会生成：

- `diagnosis_summary.json`
- `diagnosis_summary.csv`

如果需要同时导出 Word 报告，可附加：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --file your_signal.csv \
  --sampling-rate 12000 \
  --rpm 1450 \
  --output-report ./single_report.docx
```

## 系统自检

在 `程序代码/` 目录执行：

```bash
../.venv/bin/python -m pump_fault_app.app.cli --self-check
```

如果你已经准备好了演示样本配置，可附加：

```bash
../.venv/bin/python -m pump_fault_app.app.cli \
  --self-check \
  --demo-config ./configs/demo_samples.json
```

演示样本配置示例见：

- `configs/demo_samples.example.json`
- `configs/demo_package.example.json`

推荐先复制为：

- `configs/demo_samples.json`
- `configs/demo_package.json`

再把其中的 `file_path` 改为你本机的实际振动样本路径。

## 独立自检脚本

在 `程序代码/` 目录执行：

```bash
./scripts/run_pump_fault_self_check.sh
```

如果要显式传入模型 bundle 和 demo config，也可以：

```bash
./scripts/run_pump_fault_self_check.sh \
  ../实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib \
  ./configs/demo_samples.json
```

命令输出为 JSON：

- 成功时 `status="diagnosed"`，进程返回码为 `0`
- 输入错误时 `status="input_error"`，进程返回码为 `1`
- 质量拒绝时 `status="rejected"`，进程返回码为 `1`
- 如果底层模型运行时出现数值告警，会写入 `runtime_warnings`

## 批量命令行运行

在 `程序代码/` 目录执行：

```bash
../.venv/bin/python -m pump_fault_app.app.batch_cli \
  --files a.csv b.csv c.csv \
  --sampling-rate 12000 \
  --rpm 1500
```

输出为批量 JSON 汇总，包含：

- `total_count`
- `success_count`
- `failure_count`
- `diagnosed_count`
- `rejected_count`
- `input_error_count`
- `summaries`

只要批量中存在任意失败项，进程返回码为 `1`。

如果每个文件的采样率、转速或列名不同，推荐使用 manifest 清单：

```bash
../.venv/bin/python -m pump_fault_app.app.batch_cli \
  --manifest manifest.csv \
  --model-bundle ../实验结果/多转速统一六分类实验V2/six_class_models/bp/bp_bundle.joblib
```

`manifest.csv` 至少需要以下列：

- `file_path`
- `sampling_rate_hz`
- `rpm`

可选列：

- `signal_column`
- `time_column`
- `device_id`
- `measurement_position`

如果需要同时导出结果文件，可附加：

```bash
../.venv/bin/python -m pump_fault_app.app.batch_cli \
  --manifest manifest.csv \
  --output-dir ./batch_outputs
```

`--output-dir` 表示导出根目录。每次运行会自动新建一个时间戳子目录，例如：

- `batch_outputs/batch_run_20260707_153000/`

该运行目录中会生成：

- `batch_diagnosis_summary.json`
- `batch_diagnosis_summary.csv`

## 当前推荐测试

在 `程序代码/` 目录执行：

```bash
../.venv/bin/python -m pytest \
  tests/test_pump_fault_app_signal_reader.py \
  tests/test_pump_fault_app_signal_quality.py \
  tests/test_pump_fault_app_preprocessing.py \
  tests/test_pump_fault_app_windowing.py \
  tests/test_pump_fault_app_feature_extraction.py \
  tests/test_pump_fault_app_prediction.py \
  tests/test_pump_fault_app_fusion.py \
  tests/test_pump_fault_app_inference.py \
  tests/test_pump_fault_app_reporting.py \
  tests/test_pump_fault_app_cli.py \
  tests/test_pump_fault_app_batch.py \
  tests/test_pump_fault_app_batch_manifest.py \
  tests/test_pump_fault_app_batch_cli.py \
  tests/test_pump_fault_app_export.py \
  tests/test_pump_fault_app_services.py \
  tests/test_pump_fault_app_ui.py \
  tests/test_pump_fault_app_config.py \
  tests/test_pump_fault_app_logging.py \
  tests/test_pump_fault_app_records.py \
  tests/test_inference_contract.py -q
```

## Run Order

1. `python -m pump_diagnosis.runner --stage inspect`
2. `python -m pump_diagnosis.runner --stage split`
3. `python -m pump_diagnosis.runner --stage features`
4. `python -m pump_diagnosis.runner --stage train`
5. `python -m pump_diagnosis.runner --stage all`

## Feature Groups

- 18 time-domain features
- 20 frequency-domain features
- 9 rotational features
- 17 wavelet-packet features
- 20 envelope features

## 结果目录

- 基础流程结果：`../实验结果/通道4六分类_基础结果/`
- 修正倍频最终结果：`../实验结果/修正倍频_六分类最终结果/`
要求：
1. 不要重写已经完成并通过测试的模块。
2. 先检查现有项目结构和接口，再进行修改。
3. 新增单元测试。
4. 所有参数从统一配置读取，禁止在函数内重复硬编码。
5. 修改后运行全部测试。
6. 汇报新增文件、修改文件、测试结果和仍存在的问题。
