# CatBoost43 V3 验收说明

正式链路为：分通道文件读取 → 严格时间对齐校验 → 质量检查 → 12000 Hz 重采样 → 5–5000 Hz 零相位带通 → 4800/2400 分窗 → 固定 43 维特征 → 通道独立 CatBoost 推理 → 窗口概率平均 → 有效通道概率平均 → UI、历史和 Word 报告。

必须验收：三文件独立输入；时间列全有或全无并逐点对齐；单通道失效隔离；全部失效明确失败；模型与 43 维顺序经 manifest 校验；四类通道图表可见；V3 历史、报告和批处理完整；部署与评估资产隔离；不冒用原 parent_group 结果。

验证命令：

```bash
python -m pump_fault_app.app.self_check
python -m pytest -q
git diff --check
```

界面还需完成真实启动和 `/_stcore/health` 健康检查。
