# 水泵故障诊断软件演示样本筛选说明

## 1. 筛选目的

本次筛选用于在不修改模型、特征、预处理参数、窗口参数和正式推理流程的前提下，选择一条更适合答辩演示的正式单文件样本。

筛选原则：

- `status = diagnosed`
- `predicted_label = expected_label`
- `confidence >= 0.90`
- `window_count = 119`
- 四类 visualization 全部可用
- Word 报告可正常导出
- 在满足以上条件下，优先比较：
  - `warning_count`
  - `probability_margin`
  - `window_consistency`
  - 类别展示价值

## 2. 筛选范围

筛选脚本基于：

- `../实验结果/多转速统一六分类实验V2/unified_six_class_dataset_raw.csv`

共筛选 48 条候选原始文件，范围为：

- `Motor-4 / 70 / 汽蚀`
- `Motor-4 / 70 / 转子不平衡`
- `Motor-4 / 70 / 联轴器不对中`
- `Motor-2 / 100 / 轴承故障`
- `Motor-2 / 100 / 松动`
- `Motor-2 / 100 / 正常`

## 3. 关键结论

- 未找到 `warning_count` 更少的样本。
- 在本次扫描的 48 条候选样本中，所有成功诊断样本的 `warning_count` 均为 `714`。
- 因此，本轮优化重点转为：
  - 提高 `confidence`
  - 提高 `probability_margin`
  - 提高 `window_consistency`
  - 保持故障类别展示更直观

## 4. Top 10 候选样本

| 排名 | 期望类别 | 预测类别 | 置信度 | 概率差 | 窗口一致率 | 告警数 | 文件路径 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 汽蚀 | 汽蚀 | 1.0000 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀5/振动_电机4_70_时域-出口汽蚀5-通道4.csv` |
| 2 | 汽蚀 | 汽蚀 | 1.0000 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀3/振动_电机4_70_时域-出口汽蚀3-通道4.csv` |
| 3 | 轴承故障 | 轴承故障 | 1.0000 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/100/泵轴承故障2/振动_电机2_100_时域-泵轴承故障2-通道4.csv` |
| 4 | 联轴器不对中 | 联轴器不对中 | 1.0000 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/平行不对中4/振动_电机4_70_时域-平行不对中4-通道4.csv` |
| 5 | 汽蚀 | 汽蚀 | 0.9999 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀4/振动_电机4_70_时域-出口汽蚀4-通道4.csv` |
| 6 | 正常 | 正常 | 0.9998 | 1.0000 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/100/正常状态1/振动_电机2_100_时域-正常状态1-通道4.csv` |
| 7 | 联轴器不对中 | 联轴器不对中 | 0.9997 | 0.9990 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/组合不对中3/振动_电机4_70_时域-组合不对中3-通道4.csv` |
| 8 | 轴承故障 | 轴承故障 | 0.9996 | 0.9990 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/100/轴承内圈故障1/振动_电机2_100_时域-轴承内圈故障1-通道4.csv` |
| 9 | 联轴器不对中 | 联轴器不对中 | 0.9992 | 0.9980 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/组合不对中1/振动_电机4_70_时域-组合不对中1-通道4.csv` |
| 10 | 联轴器不对中 | 联轴器不对中 | 0.9989 | 0.9980 | 1.0000 | 714 | `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/平行不对中3/振动_电机4_70_时域-平行不对中3-通道4.csv` |

## 5. 最终推荐

推荐主演示样本：

- 类别：`汽蚀`
- 文件：`/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀5/振动_电机4_70_时域-出口汽蚀5-通道4.csv`

推荐原因：

- 属于优先展示的故障类
- `status = diagnosed`
- `predicted_label = expected_label`
- `confidence ≈ 1.0000`
- `probability_margin = 1.0000`
- `window_consistency = 1.0000`
- `window_count = 119`
- visualization 四类图全部可用
- Word 报告可正常导出

## 6. 备用样本

保留以下备用样本：

1. 上一轮已验证的汽蚀样本
   - `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70/出口汽蚀2/振动_电机4_70_时域-出口汽蚀2-通道4.csv`
2. 高置信度轴承故障样本
   - `/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/100/泵轴承故障2/振动_电机2_100_时域-泵轴承故障2-通道4.csv`
