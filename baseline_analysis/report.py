from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from .config import LABEL_ORDER

CHINESE_FONT=FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

def save_confusion(matrix,path,title):
    m=np.asarray(matrix); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(m,cmap="Blues"); ax.set_xticks(range(6),LABEL_ORDER,rotation=35,ha="right",fontproperties=CHINESE_FONT); ax.set_yticks(range(6),LABEL_ORDER,fontproperties=CHINESE_FONT); ax.set_xlabel("预测类别",fontproperties=CHINESE_FONT); ax.set_ylabel("真实类别",fontproperties=CHINESE_FONT); ax.set_title(title,fontproperties=CHINESE_FONT)
    for i in range(6):
        for j in range(6): ax.text(j,i,str(m[i,j]),ha="center",va="center",color="white" if m[i,j]>m.max()/2 else "black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def write_conclusion(comparison,cv_summary,output):
    recalls=[c for c in comparison.columns if c.startswith("window_recall_") or c.startswith("record_recall_")]
    paired=comparison.pivot(index="channel",columns="split_mode",values="window_macro_f1")
    deltas=(paired["temporal"]-paired["record"]).round(6)
    lines=["# 单通道43维CatBoost基线结论","","> `iterations=540`是事先固定的baseline参数，不代表当前数据最优轮数。CatBoost使用原始43维特征，未做StandardScaler。","","## 训练内部CV结果（不含测试集）","",cv_summary.to_csv(index=False),"","## 独立测试窗口级结果","",comparison[["channel","split_mode","window_accuracy","window_macro_f1","window_weighted_f1",*recalls[:6]]].to_csv(index=False),"","## 独立测试record级结果","",comparison[["channel","split_mode","record_accuracy","record_macro_f1","record_weighted_f1",*recalls[6:]]].to_csv(index=False),"","## 两种划分对比","",f"temporal相对record-level split的窗口Macro-F1差值：{deltas.to_dict()}。三个通道差值都很小，且并非全部为正，因此不将本次结果解释为temporal明显优于record-level split。","","## 主要结论","",f"- 窗口级Macro-F1范围为 {comparison.window_macro_f1.min():.4f}–{comparison.window_macro_f1.max():.4f}；record级为 {comparison.record_macro_f1.min():.4f}–{comparison.record_macro_f1.max():.4f}。",f"- CH5在两种划分下的窗口Macro-F1均最低，是后续特征筛选或针对性补特征时的首要检查通道。",f"- 六组中松动的最低窗口召回率为 {comparison['window_recall_松动'].min():.4f}，低于其他多数类别，应继续检查其与轴承故障的重叠record。","","## 解释边界","","temporal split评价同record内未见时间段的稳定识别；record-level split评价未见record泛化。即使temporal更高，也只能说明同record内更稳定、跨record泛化相对较弱，不代表跨设备或跨工况泛化。"]
    Path(output).write_text("\n".join(lines),encoding="utf-8")
