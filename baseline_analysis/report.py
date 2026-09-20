from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import LABEL_ORDER

def save_confusion(matrix,path,title):
    m=np.asarray(matrix); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(m,cmap="Blues"); ax.set_xticks(range(6),LABEL_ORDER,rotation=35,ha="right"); ax.set_yticks(range(6),LABEL_ORDER); ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别"); ax.set_title(title)
    for i in range(6):
        for j in range(6): ax.text(j,i,str(m[i,j]),ha="center",va="center",color="white" if m[i,j]>m.max()/2 else "black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def write_conclusion(comparison,cv_summary,output):
    lines=["# 单通道43维CatBoost基线结论","","> 540轮是事先固定的baseline参数，不代表当前数据最优轮数。","","## 训练内部CV结果","",cv_summary.to_csv(index=False),"","## 独立测试窗口级结果","",comparison[["channel","split_mode","window_accuracy","window_macro_f1","window_weighted_f1"]].to_csv(index=False),"","## 独立测试record级结果","",comparison[["channel","split_mode","record_accuracy","record_macro_f1","record_weighted_f1"]].to_csv(index=False),"","## 解释边界","", "temporal split评价同record内未见时间段的稳定识别；record split评价未见record泛化。temporal更高时，只能说明同record内更稳定、跨record泛化相对较弱，不代表跨设备或跨工况泛化。"]
    Path(output).write_text("\n".join(lines),encoding="utf-8")
