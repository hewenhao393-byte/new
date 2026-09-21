import json
from pathlib import Path

import pandas as pd

from .acceptance import accept_feature_tables, write_acceptance
from .config import FEATURE_COLUMNS, LABEL_ORDER
from .correlation import analyze_correlations
from .modeling import train_baseline
from .quality import class_separation, describe_features, pairwise_effects, record_variability
from .report import save_confusion, write_conclusion


def _write_quality(tables, root):
    out=root/"feature_quality"; out.mkdir()
    dimensions=["label","motor","rpm","condition","state","severity"]
    for mode,channels in tables.items():
        training="train_dev" if mode=="record" else "train"
        for channel,data in channels.items():
            for subset,frame in [("train",data[data.split.eq(training)]),("test",data[data.split.eq("test")])]:
                for dim in dimensions:
                    describe_features(frame,FEATURE_COLUMNS,[dim]).to_csv(out/f"{mode}_ch{channel}_{subset}_by_{dim}.csv",index=False,encoding="utf-8-sig")
            train=data[data.split.eq(training)]
            pairwise_effects(train,FEATURE_COLUMNS,"label","正常","联轴器不对中").to_csv(out/f"{mode}_ch{channel}_train_normal_vs_misalignment.csv",index=False,encoding="utf-8-sig")
            pairwise_effects(train,FEATURE_COLUMNS,"label","松动","轴承故障").to_csv(out/f"{mode}_ch{channel}_train_looseness_vs_bearing.csv",index=False,encoding="utf-8-sig")
            record_variability(train,FEATURE_COLUMNS).to_csv(out/f"{mode}_ch{channel}_train_record_variability.csv",index=False,encoding="utf-8-sig")
            class_separation(train,FEATURE_COLUMNS).to_csv(out/f"{mode}_ch{channel}_train_class_separation.csv",index=False,encoding="utf-8-sig")


def _write_correlations(tables,root):
    out=root/"correlations"; out.mkdir(); all_pairs=[]
    for mode,channels in tables.items():
        training="train_dev" if mode=="record" else "train"
        for channel,data in channels.items():
            result=analyze_correlations(data,FEATURE_COLUMNS,training)
            result["pearson"].to_csv(out/f"{mode}_ch{channel}_pearson_train.csv",encoding="utf-8-sig")
            result["spearman"].to_csv(out/f"{mode}_ch{channel}_spearman_train.csv",encoding="utf-8-sig")
            pairs=result["pearson_pairs"].copy(); pairs.insert(0,"channel",channel); pairs.insert(0,"split_mode",mode); all_pairs.append(pairs)
    combined=pd.concat(all_pairs,ignore_index=True) if all_pairs else pd.DataFrame()
    combined.to_csv(out/"pearson_high_correlation_pairs.csv",index=False,encoding="utf-8-sig")
    combined.to_csv(out/"suggested_feature_review_list.csv",index=False,encoding="utf-8-sig")


def _recalls(metrics):
    return {f"recall_{label}":metrics["classification_report"][label]["recall"] for label in LABEL_ORDER}


def run_pipeline(source_root,output_root):
    root=Path(output_root)
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    acceptance,tables=accept_feature_tables(source_root); write_acceptance(acceptance,root/"feature_acceptance")
    _write_quality(tables,root); _write_correlations(tables,root)
    models=root/"models"; models.mkdir(); comparison=[]; cv_frames=[]
    for mode in ("record","temporal"):
        for channel in (3,4,5):
            run_out=models/mode/f"ch{channel}"; run_out.parent.mkdir(parents=True,exist_ok=True)
            result=train_baseline(tables[mode][channel],mode,channel,run_out,run_cv=True)
            wm,rm=result["window"],result["record"]
            row={"channel":f"CH{channel}","split_mode":mode,"window_accuracy":wm["accuracy"],"window_macro_f1":wm["macro_f1"],"window_weighted_f1":wm["weighted_f1"],"record_accuracy":rm["accuracy"],"record_macro_f1":rm["macro_f1"],"record_weighted_f1":rm["weighted_f1"],**{f"window_{k}":v for k,v in _recalls(wm).items()},**{f"record_{k}":v for k,v in _recalls(rm).items()}}
            comparison.append(row)
            cv=result["cv"].copy(); cv.insert(0,"channel",f"CH{channel}"); cv.insert(1,"split_mode",mode); cv_frames.append(cv)
            for level,metrics in [("window",wm),("record",rm)]:
                pd.DataFrame(metrics["confusion_matrix"],index=LABEL_ORDER,columns=LABEL_ORDER).to_csv(run_out/f"{level}_confusion_matrix.csv",encoding="utf-8-sig")
                save_confusion(metrics["confusion_matrix"],run_out/f"{level}_confusion_matrix.png",f"CH{channel} {mode} {level}")
    comparison=pd.DataFrame(comparison); comparison.to_csv(root/"split_comparison.csv",index=False,encoding="utf-8-sig")
    cv=pd.concat(cv_frames,ignore_index=True); cv.to_csv(root/"internal_cv_all_folds.csv",index=False,encoding="utf-8-sig")
    cv_summary=cv.groupby(["channel","split_mode"],as_index=False).agg({"window_accuracy":["mean","std"],"window_macro_f1":["mean","std"],"record_accuracy":["mean","std"],"record_macro_f1":["mean","std"]}); cv_summary.columns=["_".join(x).rstrip("_") for x in cv_summary.columns]; cv_summary.to_csv(root/"internal_cv_summary.csv",index=False,encoding="utf-8-sig")
    write_conclusion(comparison,cv_summary,root/"conclusion.md")
    (root/"run_manifest.json").write_text(json.dumps({"source_root":str(Path(source_root).resolve()),"runs":6,"iterations":540,"iterations_note":"fixed baseline parameter; not data-optimal","formal_correlation_rule":"training-only Pearson abs(r)>=0.95","spearman_role":"supplementary only"},ensure_ascii=False,indent=2),encoding="utf-8")
    return comparison
