# 水泵故障诊断软件 CatBoost43 V3 重构设计

## 1. 目标与边界

本次在独立分支 `codex/pump-app-catboost43-v3` 中，将现有水泵故障诊断软件的正式推理链路从“21维特征 + BP六分类”替换为“43维多域特征 + CH3/CH4/CH5独立CatBoost六分类 + 多窗口概率平均 + 有效通道等权概率融合”。

本轮只完成正式部署模型训练、1～3通道推理、单文件与批量入口、历史记录、结果页、可视化和Word报告的V3全链路升级。不实现学习权重、类别级权重、129维联合模型、SHAP、OOD检测、自动重训练、自动替换模型或parent-group适用性评分。

原parent_group严格泛化结果、评估模型、manifest和指标保持不变。V3部署模型使用全部验收合格数据重新训练，只用于软件推理，不继承或重新声称原parent_group严格泛化性能。后续泛化评价必须使用新的独立数据。

## 2. 分支、基线与删除原则

- 当前完整软件状态已保存在基线提交 `3bd3a3c`。
- V3工作在独立分支 `codex/pump-app-catboost43-v3`，不直接修改原稳定分支。
- 所有正式链路删除均在V3分支发生，并可通过Git回滚。
- 不删除历史实验结果、parent_group评估产物、论文材料、测试数据、旧历史数据库或无关文件。
- 位于历史实验结果目录中的旧BP bundle仅解除软件引用，不删除。
- 实际删除前再次输出Git删除清单并取得用户确认。

## 3. 唯一正式V3契约

软件包内只保留一个正式契约 `FORMAL_V3_CONTRACT`：

| 项目 | 固定值 |
|---|---|
| contract | `formal-V3-catboost43` |
| software version | `pump-fault-app-v3` |
| model version | `catboost43-six-class-v3` |
| feature version | `43-feature-v1` |
| target sampling rate | 12000 Hz |
| bandpass | 5～5000 Hz，零相位 |
| window size | 4800点 |
| step size | 2400点 |
| wavelet | db6 |
| wavelet packet level | 3 |
| feature count | 43 |
| model type | CatBoost |
| channels | CH3、CH4、CH5 |

内部类别顺序固定为：`正常、转子不平衡、联轴器不对中、松动、轴承故障、汽蚀`。界面、历史和报告把内部“松动”显示为“机械松动”，不改变模型类别或概率列顺序。

所有预处理、分窗、特征、预测、融合、界面和报告参数都从这一契约读取，不允许各模块分别写死正式参数。

## 4. 43维特征契约

正式43维特征直接复用已验收实现，固定顺序为：

1. `rms`
2. `std`
3. `peak_to_peak`
4. `skewness`
5. `kurtosis`
6. `crest_factor`
7. `impulse_factor`
8. `clearance_factor`
9. `shape_factor`
10. `rot_1x_energy_ratio`
11. `rot_2x_energy_ratio`
12. `rot_3x_energy_ratio`
13. `rot_2x_1x_ratio`
14. `rot_3x_1x_ratio`
15. `harmonic_energy_ratio_1x_5x`
16. `harmonic_energy_ratio_3x_5x`
17. `rot_2x_harmonic_ratio`
18. `rot_05x_1x_ratio`
19. `noninteger_harmonic_energy_ratio`
20. `spectral_entropy`
21. `spectral_flatness`
22. `spectral_centroid`
23. `spectral_bandwidth`
24. `band_energy_5_300_ratio`
25. `band_energy_300_1000_ratio`
26. `band_energy_1000_3000_ratio`
27. `band_energy_3000_5000_ratio`
28. `high_low_energy_ratio`
29. `wp_energy_ratio_0`
30. `wp_energy_ratio_1`
31. `wp_energy_ratio_2`
32. `wp_energy_ratio_3`
33. `wp_energy_ratio_4`
34. `wp_energy_ratio_5`
35. `wp_energy_ratio_6`
36. `wp_energy_ratio_7`
37. `wp_energy_entropy`
38. `env_kurtosis`
39. `env_crest_factor`
40. `env_spectral_entropy`
41. `env_peak_energy_ratio`
42. `env_peak_concentration`
43. `env_peak_count`

分组为时域9维、阶次/谐波10维、频谱形状4维、频带能量5维、小波包9维和包络6维。启动时同时校验特征数量、名称和顺序；任一不一致都拒绝正式推理。每个通道独立生成43维，不构建129维拼接向量。

## 5. 部署模型训练与模型资产隔离

### 5.1 唯一全量训练源

使用以下三张已验收表：

- `实验结果/单通道43维_两种划分_20260920/file_split/features_ch3.csv`
- `实验结果/单通道43维_两种划分_20260920/file_split/features_ch4.csv`
- `实验结果/单通道43维_两种划分_20260920/file_split/features_ch5.csv`

三张表代表同一批1845条记录。训练时去除`subset`等元数据列并使用全部合格行；不再叠加`temporal_split`表，防止同一窗口重复训练。训练前校验源文件SHA-256、43维列、重复窗口键、三通道兄弟记录对齐、六类集合、NaN/Inf及类别分布。

### 5.2 固定P1@500参数

三个通道统一使用：

```text
loss_function = MultiClass
iterations = 500
depth = 8
learning_rate = 0.05
l2_leaf_reg = 100
random_strength = 5
rsm = 0.7
auto_class_weights = SqrtBalanced
random_seed = 2026
allow_writing_files = false
verbose = false
```

P1@500只描述此前预定义候选范围内选定的统一正式参数，不表述为全局最优参数。

### 5.3 资产目录

部署资产位于：

```text
程序代码/models/catboost43_v3/
├── ch3.cbm
├── ch4.cbm
├── ch5.cbm
└── manifest.json
```

manifest记录训练表来源及哈希、43维顺序、内部六类顺序、完整参数、每个模型哈希、训练行数、记录数、类别分布、`training_scope=all_accepted_data`和`intended_use=software_inference_only`，并明确声明不携带parent_group评估指标。

parent_group评估模型继续保存在原实验目录，由原manifest管理。训练脚本不得读取、覆盖或改写该目录。

## 6. 三文件输入契约

每次诊断允许分别输入CH3、CH4、CH5中的1～3个文件。每个通道输入包含通道标识、文件路径、信号列和可选时间列；原始采样率与rpm是本次诊断的公共参数。

严格规则如下：

- 通道标识只能是CH3、CH4、CH5且不可重复。
- 多文件信号长度必须完全相同。
- 时间列必须全部提供或全部不提供。
- 全部提供时间列时逐点校验时间轴一致，并验证时间间隔与公共采样率相容。
- 全部不提供时间列时按公共采样率生成时间轴。
- 部分文件有时间列、部分没有时直接拒绝。
- 不自动截断、补齐、插值或偏移对齐。

## 7. 数据结构

### 7.1 请求

`ChannelInput`保存`channel`、`file_path`、`signal_column`和`time_column`。`MultiChannelInferenceRequest`保存1～3个`ChannelInput`、公共`sampling_rate_hz`、公共`rpm`和可选模型目录。

### 7.2 窗口结果

`WindowDiagnosisResult`保存窗口序号、起止索引、内部预测类别和固定六类顺序的概率。旧`confidence`字段不再进入V3结构；页面需要时由概率向量计算“最大模型概率”。

### 7.3 通道结果

`ChannelDiagnosisResult`至少保存：

- channel；
- `valid`或`invalid`状态；
- failure_stage与failure_message；
- predicted_label与class_probabilities；
- window_predictions、window_count、valid_window_count和window_consistency；
- quality_report、visualization和warnings。

valid结果必须具有六类概率和类别；invalid结果的类别和概率必须为空，但保留质量报告和失败原因。

### 7.4 融合结果

`MultiChannelDiagnosisResult`至少保存：

- `diagnosed`或`failed`状态；
- input_channels、valid_channels和invalid_channels；
- 全部channel_results；
- predicted_label与fused_probabilities；
- agreement_level、agreement_count和valid_channel_count；
- 三项版本号与总体warnings。

全部通道无效时，状态为failed，最终类别和融合概率为空，但保留各通道失败信息。

## 8. 推理与融合流程

每个通道独立执行：

```text
原始信号
→ 质量检查
→ 去均值
→ 抗混叠重采样到12 kHz
→ 5～5000 Hz零相位带通
→ 4800/2400滑窗
→ 固定43维特征
→ 对应通道CatBoost
→ 每窗口六类概率
→ 概率算术平均
→ 通道级六类概率
```

通道内融合为`P_channel = mean(P_window)`，最终类别为`argmax(P_channel)`。

多通道融合只接收valid通道：

- 1通道：`P_final = P_channel`；
- 2通道：`P_final = (P_a + P_b) / 2`；
- 3通道：`P_final = (P3 + P4 + P5) / 3`。

禁止硬标签多数投票和灰度质量加权。某通道全零、NaN/Inf、严重削顶、长度不足或无法生成特征时标记invalid并剔除；全部invalid时不输出故障类别。

通道一致性规则：

- 1个有效通道：保存`1/1`，不评价跨通道一致性；
- 所有有效通道同类：高一致；
- 3个有效通道中2个同类：中等一致；
- 2个有效通道不同或3个有效通道全部不同：低一致。

## 9. 分层架构

```text
formal_contract + diagnosis_models
          │
          ├── io + quality + preprocessing + windowing
          │                         │
          │                 catboost43_features
          │                         │
          │                 channel_predictor
          │                         │
          │                   window_fusion
          │                         │
          └────────────── channel_inference
                                    │
                         multichannel_inference
                                    │
                           channel_fusion
                                    │
                     ┌──────────────┼──────────────┐
                  services         batch         self-check
                     │
          ┌──────────┼──────────┐
          UI       history     reporting
                               │
                           Word报告
```

UI、历史和报告只读取统一结果对象，不能直接调用特征提取、模型预测或重新计算融合结论。单文件和批量必须调用同一个`run_multichannel_inference()`。

## 10. UI、历史和报告

### 10.1 结果页

第一层显示最终融合结果：最终类别、模型输出概率、六类融合概率、有效通道数、通道一致性和总体告警。不得使用“诊断置信度”。

第二层显示各通道：通道、类别、最大模型概率、窗口一致率、有效窗口数和质量状态。多通道可视化允许切换查看每个通道的时域波形、频谱、包络谱和小波包能量。

### 10.2 首页与系统说明

首页只展示43维多域振动特征、CatBoost六分类、12 kHz、5～5000 Hz、4800/2400、CH3/CH4/CH5单通道与多通道诊断、多窗口概率融合和多通道等权概率融合。系统说明明确三个通道由三个独立模型完成诊断，不描述为129维单模型。

### 10.3 历史记录

使用V3独立表或数据库，不强制迁移旧BP历史。保存时间、文件名、输入通道、公共采样率、rpm、各通道结果与六类概率、最终融合结果与概率、窗口一致率、通道一致性及三项版本号。

### 10.4 Word报告

报告包含基本信息、输入通道、信号质量、各通道结果、多通道融合结果、六类融合概率、窗口一致性、通道一致性、图形分析和版本契约。删除当前报告中的21维、BP、formal-V2及旧窗口参数表述。

### 10.5 持续优化

第一阶段从导航移除模型持续优化页面，并删除旧BP候选训练、样本快照和批准逻辑，防止旧训练代码操作V3正式模型。旧数据库文件保留。

## 11. 文件变更范围

### 11.1 计划删除

- `pump_fault_app/feature_extraction/extractor.py`
- `pump_fault_app/prediction/predictor.py`
- `pump_fault_app/fusion/probability_fusion.py`
- `pump_fault_app/inference/service.py`
- `pump_fault_app/model_training/`
- `pump_fault_app/model_registry/`
- `pump_fault_app/sample_repository/`
- `pump_fault_app/ui/pages/model_optimization.py`
- `tests/test_pump_fault_app_model_optimization.py`

### 11.2 计划新增

- `pump_fault_app/domain/diagnosis_models.py`
- `pump_fault_app/feature_extraction/catboost43_features.py`
- `pump_fault_app/prediction/catboost_loader.py`
- `pump_fault_app/prediction/channel_predictor.py`
- `pump_fault_app/fusion/window_fusion.py`
- `pump_fault_app/fusion/channel_fusion.py`
- `pump_fault_app/inference/channel_inference.py`
- `pump_fault_app/inference/multichannel_inference.py`
- `pump_fault_app/history/v3_models.py`
- `pump_fault_app/history/v3_repository.py`
- `models/catboost43_v3/ch3.cbm`
- `models/catboost43_v3/ch4.cbm`
- `models/catboost43_v3/ch5.cbm`
- `models/catboost43_v3/manifest.json`
- `tools/train_catboost43_v3_deployment.py`
- `tests/test_pump_fault_app_v3_models.py`
- `tests/test_pump_fault_app_channel_fusion.py`
- `tests/test_pump_fault_app_multichannel_input.py`
- `tests/test_catboost43_v3_deployment_training.py`
- `docs/pump_fault_app_v3_model_provenance.md`

### 11.3 计划修改

修改`domain/formal_contract.py`、`version.py`、其余domain模块、预处理、分窗、质量、可视化、services、batch、history、reporting、export、UI、presentation、CLI、bootstrap、self-check、配置、包导出、现有应用测试和当前有效用户文档。旧`docs/superpowers/specs/`与`plans/`作为历史记录不回写。

## 12. 测试与验收

所有行为变更遵循测试先行。至少覆盖：V3参数、43维数量/名称/顺序、三模型加载、类别重排、单通道推理、双/三通道融合、无效通道剔除、全部无效、概率和为1、窗口融合、通道一致性、三文件严格输入、单文件服务、批量服务、历史版本字段、Word报告V3字段、旧BP/V2字符串与导入扫描。

端到端样例至少包括：仅CH3、仅CH4、CH3+CH4、CH3+CH4+CH5、一个无效通道、三个通道全部无效、正常样本和至少一个故障样本。验收链路为：输入→预处理→43维→CatBoost→窗口融合→通道融合→UI→历史→Word报告。

完成前运行目标测试、全量应用测试、`git diff --check`、模型manifest与文件哈希复核、删除清单复核及页面级验证。未经用户要求不合并V3分支。

## 13. 风险控制

- 43维算法漂移：以验收特征表和固定窗口逐列对照。
- CatBoost类别顺序漂移：读取`classes_`并显式映射到契约顺序。
- 部署/评估混用：分目录、分manifest、分用途字段，部署报告不写评估指标。
- 多文件伪对齐：严格校验长度和时间轴，禁止自动修复。
- 批量旁路旧链路：单文件与批量共用同一V3服务。
- 无效通道误融合：融合入口只接受合法valid概率向量。
- 旧持续优化误操作：移除导航与旧训练入口。
- 现版软件丢失：以`3bd3a3c`保全并在独立分支工作。
- 删除扩大：删除前输出Git清单并再次取得用户确认。
