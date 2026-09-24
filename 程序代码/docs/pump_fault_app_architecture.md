# CatBoost43 V3 软件架构

```text
CH3文件 → 预处理/分窗/43维 → CH3 CatBoost ─┐
CH4文件 → 预处理/分窗/43维 → CH4 CatBoost ─┼→ 有效通道概率平均 → 六分类
CH5文件 → 预处理/分窗/43维 → CH5 CatBoost ─┘
```

`domain/` 冻结 V3 契约；`io/` 负责独立读取和对齐；`quality/`、`preprocessing/`、`windowing/` 负责信号链；`feature_extraction/` 生成固定 43 维；`prediction/` 校验 manifest 并加载三个模型；`inference/` 和 `fusion/` 完成独立推理与概率融合；`services/` 编排批量、历史和报告；`ui/` 提供三文件输入和四类图表。

`models/catboost43_v3/` 仅放正式部署资产。评估模型和 parent_group 结果不在部署加载路径中，部署模型不继承评估模型的泛化声明。
