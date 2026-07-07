# 通道 4 水泵六分类故障诊断

在本文件所在的 `程序代码/` 目录运行以下命令。

如果你想先看“这套代码每个模块是干什么的”，先看这份总说明：

- [docs/pump_fault_app_walkthrough.md](/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_walkthrough.md:1)

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

软件正式推理只允许使用 `pump_diagnosis/inference_contract.py` 中冻结的 V2 契约：

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
