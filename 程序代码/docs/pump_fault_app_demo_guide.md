# CatBoost43 V3 演示指南

启动：

```bash
cd /Users/hewenhao/.codex/worktrees/pump-app-catboost43-v3/特征提取/程序代码
/Users/hewenhao/Documents/特征提取/.venv/bin/streamlit run pump_fault_app/ui/streamlit_app.py
```

在“多通道诊断”分别上传 CH3、CH4、CH5 文件，填写信号列；时间列要么全填要么全不填。填写公共采样率和 rpm 后开始诊断。依次展示最终类别、有效通道、一致性、融合概率、通道结果，以及所选通道的四类图表，最后下载 Word 报告并查看历史。

讲解时明确：三个部署模型只用于软件推理，不代表重新获得原 parent_group 严格泛化性能。
