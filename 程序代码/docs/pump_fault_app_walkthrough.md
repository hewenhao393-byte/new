# CatBoost43 V3 代码导览

1. `app/cli.py`、`ui/streamlit_app.py` 接收请求。
2. `domain/formal_contract.py` 冻结 12000 Hz、5–5000 Hz、4800/2400、db6 level 3、43 维和六类顺序。
3. `io/` 独立读取三通道并严格对齐。
4. `quality/`、`preprocessing/`、`windowing/` 完成信号链。
5. `feature_extraction/catboost43_features.py` 提取训练一致的 43 维。
6. `prediction/` 校验 deployment manifest 并按通道加载模型。
7. `inference/`、`fusion/` 完成通道内和通道间概率平均。
8. `inference/visualization.py` 生成四类图表。
9. `history/`、`reporting/` 保存 V3 历史并导出 Word。
10. `ui/pages/v3_pages.py` 提供三文件输入与结果页面。

评估资产与部署资产分离；原 parent_group 性能只对应原评估模型，全量重训模型不用于重新声称该性能。
