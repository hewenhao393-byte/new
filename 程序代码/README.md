# 水泵故障诊断软件（CatBoost43 V3）

正式软件采用固定 43 维特征与三个独立 CatBoost 六分类模型。CH3、CH4、CH5 分别读取一个文件、分别完成质量检查和推理，再对有效通道的六类概率做等权算术平均。

## 正式推理契约

- 输入：CH3、CH4、CH5 中任选 1–3 个通道，每个通道单独上传 CSV/TXT
- 公共参数：原始采样率、转速 rpm
- 时间列：所有已选通道全部提供或全部不提供；提供时必须逐点对齐
- 12000 Hz；5–5000 Hz 零相位带通；4800 点窗口、2400 点步长
- 固定顺序 43 维特征，db6 三层小波包；包络带通 1000–5000 Hz
- CH3、CH4、CH5 各自使用独立 CatBoost 模型
- 先平均通道内窗口概率，再平均有效通道概率
- 六类：正常、转子不平衡、联轴器不对中、松动、轴承故障、汽蚀

## 模型边界

部署模型位于 `models/catboost43_v3/`，由全部验收合格数据按 P1@500 参数分别重训，仅用于软件推理。部署 manifest 与评估资产分开管理。

原有 parent_group 严格泛化结果及其评估模型保持不变。不得把全量重训后的部署模型重新表述为具有原 parent_group 泛化性能；后续泛化评价应使用新的独立数据。

## 启动界面

```bash
cd /Users/hewenhao/.codex/worktrees/pump-app-catboost43-v3/特征提取/程序代码
/Users/hewenhao/Documents/特征提取/.venv/bin/streamlit run pump_fault_app/ui/streamlit_app.py
```

浏览器打开终端显示的地址（通常为 `http://localhost:8501`）。在“多通道诊断”页分别上传各通道文件，填写各自信号列和可选时间列，再填写公共采样率与转速。

## 命令行

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pump_fault_app.app.cli \
  --channel CH3 /path/ch3.csv signal \
  --channel CH4 /path/ch4.csv signal \
  --sampling-rate 12000 --rpm 1500
```

第三个通道按同样格式增加 `--channel CH5 ...`。如使用时间列，在每个 `--channel` 参数末尾都填写对应时间列。

## 验证

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pump_fault_app.app.self_check
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest -q
```

训练来源、数据量、参数和哈希见 `docs/pump_fault_app_v3_model_provenance.md`。
