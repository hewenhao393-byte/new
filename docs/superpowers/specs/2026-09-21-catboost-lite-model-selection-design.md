# 43维单通道CatBoost节省计算版正式模型选择设计

## 目标

在不使用测试集选参的前提下，为CH3、CH4、CH5和record-level/temporal两种划分确定一套统一、稳定、适合论文解释的CatBoost参数和迭代轮数，然后训练六个最终模型并产出论文实验数据。

## 固定合同

- 仅使用原始43维特征，不删减、不扩维。
- record-level split和temporal split保持不变。
- 不使用StandardScaler。
- `loss_function=MultiClass`、`auto_class_weights=SqrtBalanced`、`random_seed=2026`。
- CH3/CH4/CH5和两种划分的六个最终模型使用同一套参数与轮数。
- 测试集仅在参数锁定后预测一次，不参与任何选择。

## 训练侧选参

仅使用record-level split的`train_dev`。按`record_id`生成一份共享的3折：

```python
StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
```

三个通道必须复用完全相同的record折索引和折哈希。

只比较：

| 候选 | depth | learning_rate | l2_leaf_reg | random_strength | rsm |
|---|---:|---:|---:|---:|---:|
| P1 | 8 | 0.05 | 100 | 5 | 0.7 |
| P2 | 6 | 0.05 | 100 | 5 | 0.7 |
| P5 | 8 | 0.05 | 200 | 5 | 0.7 |

每个折模型训练到500轮，仅在`[50,100,150,200,300,400,500]`获取分阶段概率。每个检查点先对验证窗口按`record_id`平均六类概率，再计算record级Macro-F1。

每个“参数+轮数”组合输出CH3、CH4、CH5的3折`mean±std`，主分数为：

```text
Score=(CH3_mean+CH4_mean+CH5_mean)/3
Stability=(CH3_std+CH4_std+CH5_std)/3
```

## 统一参数选择

先找到Score最高组合。将与最高分差值不超过0.002的组合视为简化候选，依次优先：

1. depth更小；
2. depth相同时轮数更少；
3. 仍相同时L2更强；
4. 仍相同时Stability更小；
5. 仍相同时按候选ID排序保证确定性。

参数锁定后不根据测试结果调整。P1/P2/P5无论差异大小都先输出结果，不自动扩展P3/P4/P6，不自动升级5折。

## 六个最终模型

锁定统一参数后，分别用各自完整训练集拟合CH3/CH4/CH5 × record/temporal共六组模型。每组保存模型、参数、特征顺序、测试窗口概率、record平均概率、特征重要性和错误明细。

评价包括窗口级和record级Accuracy/Macro-F1/Weighted-F1、六类Precision/Recall/F1、两级混淆矩阵、正常误报率，以及正常→不对中、松动→轴承、轴承→松动的窗口和record清单。错误按motor、rpm、condition、state、severity聚合。

## 输出和验收

输出到`实验结果/正式参数选择_43维_三折节省计算版_20260921/`，目录必须颈先不存在，通过唯一暂存目录原子发布，不覆盖或删除现有结果。

验收重新检查六张源表、共享3折、27个CV训练组合的结果完整性、选择规则、6个最终模型重载概率、指标重算、混淆矩阵、概率和record计数。报告将训练侧CV、测试窗口级和测试record级严格分开。
