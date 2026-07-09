# 水泵六分类故障诊断软件骨架设计

**目标**

在现有实验代码之外，新建一个面向后续“软件/系统”开发的独立包，用于统一管理六分类故障诊断软件的固定参数、数据结构、日志能力和模块接口。当前阶段不实现模型训练、推理算法和界面。

**设计原则**

- 与现有 `pump_diagnosis` 实验代码分层，避免训练脚本与软件骨架耦合。
- 固定参数集中管理，确保后续特征提取、推理和界面读取同一配置源。
- 先定义清晰的数据结构和接口，再在后续阶段接入具体实现。
- 保持与现有六分类实验的标签顺序和 21 维最终特征名单一致。

**固定配置**

- 目标采样率：`12000 Hz`
- 带通范围：`10~5000 Hz`
- 窗口长度：`2400`
- 步长：`1200`
- 包络频带：`2000~5000 Hz`
- 小波包：`db6`，三层
- 六类标签：`正常`、`转子不平衡`、`联轴器不对中`、`松动`、`轴承故障`、`汽蚀`
- 21 维最终特征名单沿用现有多转速统一六分类实验 V2

**包结构**

- `程序代码/pump_fault_app/config/`
  - `defaults.py`：固定常量与默认值
  - `schema.py`：配置 dataclass
  - `loader.py`：统一构建和序列化配置
- `程序代码/pump_fault_app/domain/`
  - `labels.py`：六类标签定义与校验
  - `features.py`：21 维特征定义与分组
  - `records.py`：样本、窗口、特征向量、诊断结果等数据结构
- `程序代码/pump_fault_app/logging/`
  - `context.py`：运行上下文字段
  - `setup.py`：日志初始化
- `程序代码/pump_fault_app/interfaces/`
  - `feature_extractor.py`
  - `classifier.py`
  - `repository.py`
- `程序代码/pump_fault_app/adapters/legacy/`
  - `contracts.py`：预留旧实验代码对接契约
- `程序代码/pump_fault_app/app/`
  - `bootstrap.py`：应用启动入口

**接口边界**

- `FeatureExtractor`：输入窗口样本和运行配置，输出 `FeatureVector`
- `Classifier`：输入 `FeatureVector`，输出 `DiagnosisResult`
- `SampleRepository`：负责样本、特征和结果的读写契约
- `LegacyFeatureTableContract`：后续将现有 `pump_diagnosis` 特征表导入软件模块时的适配边界

**日志方案**

- 提供统一 `setup_logging()` 入口
- 同时支持控制台输出和文件输出
- 使用统一格式携带 `run_id`、`component`、`label_space` 等上下文字段
- `bootstrap` 入口默认写入 `程序代码/runtime_logs/`

**验证范围**

- 配置契约测试：固定参数、标签顺序、21 维特征顺序
- 数据结构测试：序列化字段与元数据约束
- 日志测试：初始化后可写出日志文件
- 启动测试：`python -m pump_fault_app.app.bootstrap` 可打印配置摘要并写日志

**非目标**

- 不实现模型训练
- 不实现推理算法
- 不实现 GUI/前端界面
- 不改动现有 `pump_diagnosis` 训练与实验流水线
