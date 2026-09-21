from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from baseline_analysis.report import save_confusion
from .config import CHECKPOINTS,LABEL_ORDER

FONT=FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

def create_reports(root,fold_scores,summary,chosen,comparison):
    root=Path(root); figures=root/"figures"; figures.mkdir()
    for candidate,g in summary.groupby("candidate"):
        fig,ax=plt.subplots(figsize=(8,5));
        for ch,color in zip((3,4,5),("#2563eb","#059669","#d97706")):
            ax.plot(g.iteration,g[f"ch{ch}_mean"],marker="o",label=f"CH{ch}",color=color)
        ax.plot(g.iteration,g.score,color="black",linewidth=2,label="三通道等权平均")
        if candidate==chosen["candidate"]: ax.axvline(chosen["iteration"],color="red",linestyle="--",label=f"选定 {int(chosen['iteration'])}轮")
        ax.set_xlabel("迭代轮数",fontproperties=FONT); ax.set_ylabel("record级 Macro-F1",fontproperties=FONT); ax.set_title(f"{candidate} 训练侧3折CV",fontproperties=FONT); ax.legend(prop=FONT); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/f"cv_{candidate}.png",dpi=180); plt.close(fig)
    lines=["# CatBoost正式参数选择与单通道结果","","> 参数仅使用record-level split的train_dev进行3折record分组CV选择，测试集未参与选参。","","## 最终统一参数","",f"- 候选：{chosen['candidate']}",f"- 迭代轮数：{int(chosen['iteration'])}",f"- 训练侧主分数：{chosen['score']:.6f}",f"- 平均折间标准差：{chosen['stability']:.6f}","- 简化优先规则：主分数距最高分不超过0.002时，依次优先浅树、少迭代、强L2、低折间波动。","","## 训练内部CV","",summary.to_csv(index=False),"","## 独立测试窗口级","",comparison[[c for c in comparison if c.startswith('window_') or c in ('channel','split_mode')]].to_csv(index=False),"","## 独立测试record级","",comparison[[c for c in comparison if c.startswith('record_') or c in ('channel','split_mode')]].to_csv(index=False)]
    (root/"conclusion.md").write_text("\n".join(lines),encoding="utf-8")

def save_run_confusions(run_dir,wm,rm,title):
    for level,m in [("window",wm),("record",rm)]:
        pd.DataFrame(m["confusion_matrix"],index=LABEL_ORDER,columns=LABEL_ORDER).to_csv(Path(run_dir)/f"{level}_confusion_matrix.csv",encoding="utf-8-sig")
        level_title="窗口级" if level=="window" else "record级"
        save_confusion(m["confusion_matrix"],Path(run_dir)/f"{level}_confusion_matrix.png",f"{title} {level_title}混淆矩阵")
