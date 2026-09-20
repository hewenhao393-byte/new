# 单通道43维特征验收、质量分析与CatBoost基线设计

## 目标与范围

以 `实验结果/单通道43维_两种划分_20260920/` 中的六张最终特征表为唯一模型输入，完成六表验收、43维特征质量分析、训练集内相关性分析、CH3/CH4/CH5在record split和temporal split下的六组CatBoost六分类基线，以及窗口级、record级评价和错误来源分析。

本阶段不做三通道融合，不修改43维特征定义，不增加新特征，不删除难样本，不修改标签，不进行大规模超参数搜索。不对CatBoost输入执行 `StandardScaler`。

## 权威数据和输出隔离

权威输入是六张 `features_ch3.csv`、`features_ch4.csv`、`features_ch5.csv`，以及对应的 `file_split.csv`和 `temporal_split.csv`。每张表必须保持原始43个特征列、元数据列和行数，分析不回写源表。

所有新产物写入 `实验结果/单通道43维_CatBoost基线_20260920/`。如目录已存在且配置或输入哈希不同，程序必须停止，不得覆盖。六张源特征表的SHA-256、特征顺序、类别顺序、划分定义和模型参数写入运行清单。

## 六表验收

验收必须重新从磁盘读取六张表，不仅依赖上一阶段的审计文件。检查项包括：

1. 每行恰好43个已定义特征，列名与顺序完全一致。
2. 43个特征中无NaN和Inf。
3. 同一划分下CH3、CH4、CH5的 `(record_id, split, window_id, start_sample, end_sample)` 完全对齐，`label`和 `rpm` 一致。
4. record split的 `train_dev record_id` 与 `test record_id` 无交集。
5. temporal split中每个record的train/test范围无重叠，guard为12000点（1.0秒），任何窗口不进入或跨越guard，训练窗口不跨 `train_block_id` 边界。
6. `wp_energy_ratio_0`至 `wp_energy_ratio_7` 的行和在容差 `1e-8` 内等于1。
7. 四个频带能量占比的行和在容差 `1e-6` 内等于1。
8. 记录实际最大误差、重复键数、缺失数、不一致数和泄漏交集数，不只输出PASS/FAIL。

验收未通过时不进入模型训练。

## 特征质量分析

分别对CH3、CH4、CH5输出特征统计，主表按split mode和训练/测试子集分开，避免把测试集分布用于训练前筛选。分析维度包括六类故障、正常与联轴器不对中、松动与轴承故障、motor、rpm、condition、state和可用severity。severity缺失保持为缺失，不推测。

每个特征至少输出 `count`、`mean`、`std`、`median`、`Q1`、`Q3`和IQR。两类差异表同时输出：

- 标准化均值差（pooled-standard-deviation effect size）。
- 中位数差及以两类合并IQR归一化的稳健差异。

跨record波动不直接把重叠窗口视为独立样本。先对每个 `record_id`的窗口求特征均值和中位数，再对record聚合值计算均值、标准差、中位数、IQR和变异程度。

## 训练集内相关性分析

record split仅使用 `split=train_dev`，temporal split仅使用 `split=train`。每个通道和每种划分单独计算43×43 Pearson相关矩阵，输出 `|r| >= 0.95` 的特征对。

本阶段不删除任何特征。建议清单以确定性规则生成：优先保留缺失率低、数值稳定、跨record波动较小且物理含义明确的特征。`rot_2x_1x_ratio`、`rot_3x_1x_ratio`、`rot_05x_1x_ratio`、`harmonic_energy_ratio_1x_5x`及其他明确阶次/谐波特征即使高相关也标记为“物理特征，人工复核”，不机械列入删除项。

## CatBoost基线协议

六组实验为CH3/CH4/CH5 × record/temporal。所有实验使用相同43维列顺序和相同固定参数：

```text
loss_function = MultiClass
iterations = 540
depth = 8
learning_rate = 0.05
l2_leaf_reg = 100
random_strength = 5
rsm = 0.7
auto_class_weights = SqrtBalanced
random_seed = 2026
allow_writing_files = False
```

不使用StandardScaler，不使用测试集早停，不根据测试结果修改轮数、类别权重、阈值或参数。

训练内部稳定性评价使用 `StratifiedGroupKFold`：

- record split：候选训练窗口为 `train_dev`，`groups=record_id`。
- temporal split：候选训练窗口为 `train`，`groups=record_id + "::" + train_block_id`，确保不同record中重名块号不会被误合并。

默认5折；如果任一类独立Group数不足5，降为3折并写入报告。内部折只报告稳定性，不进行参数搜索。最终模型使用所有训练窗口拟合，测试集仅预测一次。

## 评价、record融合和错误分析

窗口级指标包括Accuracy、Macro-F1、Weighted-F1、每类Precision/Recall/F1和6×6混淆矩阵。record级对每个 `record_id` 内的六类窗口预测概率取平均，以平均概率最大的类别作为最终预测，再计算同样指标和混淆矩阵。

额外输出正常误报率、联轴器不对中召回率、松动召回率、轴承故障召回率和汽蚀召回率。正常误报率定义为真实正常样本中被预测为任一故障的比例，窗口级与record级分别计算。

保存每个测试窗口和record的真实标签、预测标签、六类概率及 `record_id/motor/rpm/condition/state/severity`。误分类按这些维度统计，单独输出正常→联轴器不对中和松动→轴承故障的record列表、错分窗数、总窗数和错分集中度。

## 特征重要性与结果对比

每个最终CatBoost模型输出全43维 `PredictionValuesChange` 特征重要性和前20名。报告标记第一特征、前5累计重要性、小波包特征在前20中的数量，以检查对单一特征或小波包节点的过度依赖。

SHAP不是完成条件。只在不改变模型且对固定种子抽样计算的资源开销可控时，输出SHAP summary；未执行时明确记录原因。

生成六行总对比表，包含通道、split mode、窗口级Macro-F1、record级Macro-F1、Accuracy、Weighted-F1和六类Recall。temporal明显高于record split时，结论限定为“同record内未见时间段识别更稳定，跨record泛化相对较弱”，不扩展为跨设备或跨工况泛化结论。

## 产物结构

```text
实验结果/单通道43维_CatBoost基线_20260920/
  run_manifest.json
  feature_acceptance/
  feature_quality/
  correlation/
  models/{record|temporal}/ch{3|4|5}/
  comparison/
  figures/confusion_matrices/
  conclusion_report.md
```

`feature_acceptance` 包含完整验收表和不一致明细；`feature_quality` 包含分组统计、类间差异和跨record波动；`correlation` 包含六张相关矩阵、高相关特征对和建议清单；每个模型目录包含 `.cbm` 模型、参数、内部CV、窗口/record预测、指标、分类报告、混淆矩阵、错误来源和特征重要性。`comparison` 包含六组总对比表和按split/channel的差异表。

## 验证策略

实现采用测试驱动。单元测试覆盖特征列契约、划分泄漏拒绝、时间范围/guard检查、三通道对齐、概率平均record融合、指标、正常误报率、相关性保护标记、误分类聚合和特征重要性顺序。先对受控小表完成端到端测试，再对六张真实表执行验收。验收通过后先运行一组基线并完成模型重载概率一致性检查，再执行其余5组。最终重新读取所有CSV/JSON/模型和PNG，独立复核总表与六组明细指标一致。
