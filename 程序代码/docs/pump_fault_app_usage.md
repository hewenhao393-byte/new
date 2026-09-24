# CatBoost43 V3 使用说明

运行 `streamlit run pump_fault_app/ui/streamlit_app.py`。在“多通道诊断”分别上传 CH3、CH4、CH5 文件；各文件独立填写信号列和可选时间列，共享采样率与 rpm。时间列必须全部提供或全部不提供并逐点对齐，软件不会截断、补齐或插值掩盖不一致。

CLI 示例：

```bash
python -m pump_fault_app.app.cli \
  --channel CH3 /path/ch3.csv signal time \
  --channel CH4 /path/ch4.csv signal time \
  --channel CH5 /path/ch5.csv signal time \
  --sampling-rate 12000 --rpm 1500
```

不使用时间列时，每个 `--channel` 都省略最后的 time 参数。有效通道分别执行 43 维 CatBoost 推理，窗口概率和通道概率依次平均；无效通道显示原因，全部无效则失败。

部署模型只用于软件推理；原 parent_group 严格泛化结果和评估模型保持独立。
