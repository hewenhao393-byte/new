# pump_fault_app 模块说明

这份文档不是给程序员看的，而是给“想知道这个软件现在到底做到哪一步、每个模块负责什么”的人看的。

你可以把 `pump_fault_app` 理解成一个已经搭好的“后端骨架 + 界面骨架”。

它现在已经能完成这些事情：

1. 读入振动数据文件
2. 检查信号是否适合诊断
3. 按正式参数做预处理
4. 按正式窗口切片
5. 提取固定 21 维特征
6. 调用正式 BP 模型做六分类判断
7. 把多个窗口的结果合成为一条记录的最终判断
8. 生成中文摘要
9. 支持单文件和批量运行
10. 支持命令行和 Streamlit 界面

---

## 1. 整个系统怎么理解

最上层可以分成 4 层：

1. 契约层
   规定这个软件必须遵守什么参数和顺序。
2. 算法执行层
   真正做读文件、滤波、分窗、特征提取、模型预测。
3. 服务层
   把很多底层步骤串成“单文件诊断”或“批量诊断”。
4. 界面层
   给用户一个按钮和页面，最终还是调用服务层。

这 4 层的关系可以简化成：

`界面层 -> 服务层 -> 算法执行层 -> 契约层`

也就是说：

- 界面不直接碰滤波和模型
- 服务层不自己重新写算法
- 算法层不随意改参数
- 所有正式参数都受契约层控制

---

## 2. 契约层：这个软件必须遵守什么

核心文件：

- [程序代码/pump_diagnosis/inference_contract.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_diagnosis/inference_contract.py:1)

这是整个软件最重要的“规矩文件”。

它固定了：

- 目标采样率：`12000 Hz`
- 带通滤波：`10~5000 Hz`
- 窗口长度：`2400`
- 步长：`1200`
- 包络频带：`2000~5000 Hz`
- 小波包：`db6`，3 层
- 六类标签顺序
- 21 维最终特征顺序
- 正式模型文件位置

它的作用不是“运行算法”，而是防止后面每个模块各写各的参数，最后软件能跑但结果不可信。

你可以把它理解成：

“这个软件的国家标准”

---

## 3. 数据读取层：先把文件读进来

核心文件：

- [程序代码/pump_fault_app/io/signal_reader.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/io/signal_reader.py:1)

这个模块只干一件事：

把用户上传的 `csv` 或 `txt` 文件，读成一条一维振动信号。

它负责：

- 判断文件是否存在
- 判断文件是不是空的
- 检查采样率和转速有没有提供
- 自动识别数值列
- 优先选 `通道4`
- 如果用户指定列名，就按指定列读

它的输出是一个 `RawSignalRecord`，可以理解成：

“一条原始振动记录对象”

这个对象里会保存：

- 文件名
- 样本点数
- 采样率
- 时长
- 转速
- 设备编号
- 测点位置
- 真正的振动数组

---

## 4. 质量检查层：先判断这条信号能不能诊断

核心文件：

- [程序代码/pump_fault_app/quality/signal_quality.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/quality/signal_quality.py:1)

这个模块负责回答：

“这条信号现在适不适合继续往下做诊断？”

它会检查：

- 有没有 NaN
- 有没有无穷大
- 采样率够不够
- 长度够不够一个正式窗口
- 是不是全 0
- 是不是常量
- 有没有大量 0
- 有没有突跳
- 有没有接近削顶

它输出的是 `SignalQualityReport`。

这个结果有两类信息：

1. `rejection_reasons`
   这是直接拒绝诊断的原因。
2. `warnings`
   这是提醒，不一定阻止诊断。

所以它的定位是：

“诊断前安检”

---

## 5. 预处理层：把原始信号变成正式信号

核心文件：

- [程序代码/pump_fault_app/preprocessing/signal_preprocessing.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/preprocessing/signal_preprocessing.py:1)

它负责做正式推理前的标准处理：

1. 去直流
2. 重采样到 `12000 Hz`
3. 做 `10~5000 Hz` 带通滤波
4. 滤波后再次去直流

它输出 `PreprocessedSignalRecord`。

这个对象的意思是：

“已经按正式软件规则预处理过的信号”

---

## 6. 分窗层：把一条长信号切成很多小窗口

核心文件：

- [程序代码/pump_fault_app/windowing/segmenter.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/windowing/segmenter.py:1)

这一步使用正式参数：

- 窗口长度：`2400`
- 步长：`1200`

它输出：

- `SignalWindow`
- `WindowingResult`

你可以把它理解成：

“把整条振动记录切成很多可以单独分析的小片段”

为什么要这样做？

因为模型不是直接看整条超长信号，而是看一个个固定长度的小窗口。

---

## 7. 特征提取层：把每个窗口变成 21 个数字

核心文件：

- [程序代码/pump_fault_app/feature_extraction/extractor.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/feature_extraction/extractor.py:1)

这是“从波形到特征”的关键层。

它会从每个窗口里提取固定 21 维特征，包括：

- 时域统计量
- 倍频相关特征
- 频谱熵/谱平坦度
- 小波包 8 个能量比例
- 包络峭度、包络峰值指标

它输出 `FeatureVector`。

这个对象可以理解成：

“模型真正要看的特征向量”

不是一整段波形，而是 21 个固定顺序的数字。

---

## 8. 模型预测层：把 21 维特征送进正式 BP 模型

核心文件：

- [程序代码/pump_fault_app/prediction/predictor.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/prediction/predictor.py:1)

这一步做两件事：

1. 读取正式模型 bundle
2. 对单个窗口做预测

这个 bundle 里包含：

- `imputer`
- `scaler`
- `model`
- `features`

意思是：

- 缺失值处理器
- 标准化器
- BP 模型本体
- 模型训练时使用的正式特征顺序

这个模块会保证：

- 输入特征名顺序必须和正式顺序完全一致
- 输出概率必须重排成正式六类顺序

输出对象是 `WindowPredictionResult`，表示：

“一个窗口的预测结果”

其中包括：

- 预测类别
- 置信度
- 六类概率

---

## 9. 融合层：把多个窗口合成一条记录的最终结果

核心文件：

- [程序代码/pump_fault_app/fusion/probability_fusion.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/fusion/probability_fusion.py:1)

模型是按窗口预测的，但用户关心的是：

“这一整条振动记录到底属于哪种故障？”

所以这里会做：

- 按正式六类顺序，对多个窗口的概率逐类求平均
- 取平均后概率最大的类别作为最终结果

输出对象是 `RecordPredictionResult`，表示：

“一整条记录的最终诊断结果”

---

## 10. 推理编排层：把前面所有步骤串成一条完整流程

核心文件：

- [程序代码/pump_fault_app/inference/service.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/inference/service.py:1)

这一步就是把前面所有底层模块串起来：

`读文件 -> 质量检查 -> 预处理 -> 分窗 -> 特征提取 -> 单窗预测 -> 多窗融合`

对外暴露的是：

- `FormalInferenceRequest`
- `FormalInferenceResult`
- `run_formal_inference(...)`

所以你可以把它理解成：

“正式单文件诊断引擎”

另外这里还做了一件很重要的事：

它会捕获模型推理阶段的运行时 warning。

这些 warning 不会直接丢失，而是进入：

- `runtime_warnings`

---

## 11. 摘要层：把底层结果翻译成软件能展示的内容

核心文件：

- [程序代码/pump_fault_app/reporting/summary.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/reporting/summary.py:1)

这个模块不是重新推理，而是把推理结果整理成：

- 用户能看懂的摘要
- 界面能直接展示的字段
- 导出文件能直接写入的字段

它输出的是 `DiagnosisSummary`。

里面包括：

- 是否成功
- 诊断状态
- 最终类别
- 置信度
- 窗口数量
- 质量信息
- 原始运行告警
- 中文结构化运行告警 `runtime_alerts`

`runtime_alerts` 的意义是：

把英文 warning 变成可以在软件里直接展示的中文提示。

---

## 12. 导出层：把结果落成文件

核心文件：

- [程序代码/pump_fault_app/export/writer.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/export/writer.py:1)

它负责把结果写成：

- JSON
- CSV

支持两类导出：

1. 单文件结果导出
2. 批量结果导出

还提供统一目录命名：

- `single_run_时间戳`
- `batch_run_时间戳`

这样多次运行不会互相覆盖。

---

## 13. 批量层：一次跑多条记录

核心文件：

- [程序代码/pump_fault_app/batch/service.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/batch/service.py:1)
- [程序代码/pump_fault_app/batch/manifest.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/batch/manifest.py:1)

批量层分两种模式：

### 模式 A：文件列表

给一批文件，整批共用：

- 采样率
- 转速
- 列名信息

### 模式 B：manifest 清单

用一个 `manifest.csv` 描述每个文件自己的：

- 文件路径
- 采样率
- 转速
- 可选列名
- 可选设备信息

这样每个文件都可以有自己的参数。

---

## 14. 服务层：给界面层一个统一入口

核心文件：

- [程序代码/pump_fault_app/services/app_service.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/services/app_service.py:1)

这是给界面层准备的，不是给算法层准备的。

它的作用是：

把“推理 + 摘要 + 导出”再往上封装一层。

对外提供两个统一入口：

### 单文件

- `run_single_diagnosis(...)`

### 批量

- `run_batch_diagnosis(...)`

所以界面层不需要自己写：

- 先调推理
- 再调摘要
- 再调导出

只要直接调 service 即可。

这就是现在整个项目的“正式上层入口”。

---

## 15. 界面层：现在已经有 Streamlit 骨架

核心文件：

- [程序代码/pump_fault_app/ui/streamlit_app.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/streamlit_app.py:1)
- [程序代码/pump_fault_app/ui/pages/single_diagnosis.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/pages/single_diagnosis.py:1)
- [程序代码/pump_fault_app/ui/pages/batch_diagnosis.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/pages/batch_diagnosis.py:1)
- [程序代码/pump_fault_app/ui/pages/report_view.py](/Users/hewenhao/Documents/特征提取/程序代码/pump_fault_app/ui/pages/report_view.py:1)

当前界面层已经做到了：

- 首页
- 单文件诊断页
- 批量诊断页
- 报告查看页

而且它遵守一个原则：

界面层只调用 `pump_fault_app.services`

不直接调用：

- 读文件模块
- 预处理模块
- 特征提取模块
- 模型预测模块

这说明架构是干净的。

---

## 16. 现在整个系统怎么跑

如果从最简角度看，系统现在有 3 种入口：

### 入口 1：命令行单文件

- `pump_fault_app.app.cli`

### 入口 2：命令行批量

- `pump_fault_app.app.batch_cli`

### 入口 3：界面

- `pump_fault_app.ui.streamlit_app`

而这三个入口的后面，最终都走到统一 service：

- `run_single_diagnosis(...)`
- `run_batch_diagnosis(...)`

所以整个软件已经不是很多零散脚本，而是已经有明确主干了。

---

## 17. 你现在最应该记住的几个文件

如果你不想看太多代码，只记这几个文件就够了：

### 1. 正式规则

- `pump_diagnosis/inference_contract.py`

看它就知道软件正式参数是什么。

### 2. 单文件正式入口

- `pump_fault_app/services/app_service.py`

里面的 `run_single_diagnosis(...)`

### 3. 批量正式入口

- `pump_fault_app/services/app_service.py`

里面的 `run_batch_diagnosis(...)`

### 4. 最终摘要结构

- `pump_fault_app/reporting/summary.py`

### 5. Streamlit 界面

- `pump_fault_app/ui/`

---

## 18. 这套代码现在还缺什么

虽然骨架已经很完整，但还不是最终成品。

目前还缺：

1. `wav` 正式读取链路还没补全
2. 界面还只是第一阶段
3. 报告页还比较轻
4. 真实模型数值稳定性还可以继续优化
5. 如果以后做桌面软件，可以把 Streamlit 换成 PyQt

---

## 19. 一句话总结

你现在这套 `pump_fault_app` 已经不是“几个实验脚本”了。

它已经变成了一套有明确分层的诊断软件后端：

- 契约固定参数
- 算法模块各司其职
- 服务层统一调度
- 界面层只负责交互

后面无论继续做：

- Streamlit 演示版
- PyQt 桌面版
- 论文答辩展示

都可以继续沿着这套结构往上搭，而不用推倒重来。
