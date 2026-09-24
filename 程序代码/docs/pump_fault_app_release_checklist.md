# CatBoost43 V3 发布检查表

- [ ] 独立 `codex/pump-app-catboost43-v3` 分支
- [ ] 三个部署模型和 deployment manifest 齐全、哈希匹配
- [ ] 部署模型与 parent_group 评估资产分目录、分 manifest
- [ ] 43 维、六类、P1@500 及信号参数固定
- [ ] 三文件输入、严格时间对齐、故障隔离通过
- [ ] 四类通道图表、历史、批量、CLI、Word 报告正常
- [ ] UI 启动及健康检查通过
- [ ] 全量测试与 `git diff --check` 通过
- [ ] 发布说明明确部署模型只用于推理，不继承原泛化结论
