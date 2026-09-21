import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from baseline_analysis.evaluation import evaluate_predictions,fuse_records
from .config import COMMON,FEATURES,LABEL_ORDER,META
from .cv import _pred_frame

def _error_source_summary(predictions,error_name,total_name):
    frames=[]
    for col in ["motor","rpm","condition","state","severity"]:
        errors=(predictions[predictions.label.ne(predictions.predicted_label)]
                .groupby(col,dropna=False).size().rename(error_name))
        totals=predictions.groupby(col,dropna=False).size().rename(total_name)
        table=pd.concat([errors,totals],axis=1).fillna(0).reset_index().rename(columns={col:"value"})
        table.insert(0,"dimension",col)
        table["error_rate"]=table[error_name]/table[total_name]
        frames.append(table)
    return pd.concat(frames,ignore_index=True)

def train_final(data,mode,channel,params,iterations,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    train_split="train_dev" if mode=="record" else "train"
    train=data[data.split.eq(train_split)].reset_index(drop=True); test=data[data.split.eq("test")].reset_index(drop=True)
    model=CatBoostClassifier(**COMMON,**params,iterations=int(iterations)); model.fit(Pool(train[FEATURES],train.label)); classes=list(model.classes_)
    prob=model.predict_proba(test[FEATURES]); pred=_pred_frame(test,prob,classes); rec=fuse_records(pred,classes)
    model.save_model(str(out/"model.cbm")); restored=CatBoostClassifier(); restored.load_model(str(out/"model.cbm")); np.testing.assert_allclose(prob,restored.predict_proba(test[FEATURES]),atol=1e-12)
    wm=evaluate_predictions(pred.label,pred.predicted_label,LABEL_ORDER); rm=evaluate_predictions(rec.label,rec.predicted_label,LABEL_ORDER)
    pred.to_csv(out/"window_predictions.csv",index=False,encoding="utf-8-sig"); rec.to_csv(out/"record_predictions.csv",index=False,encoding="utf-8-sig")
    (out/"window_metrics.json").write_text(json.dumps(wm,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"record_metrics.json").write_text(json.dumps(rm,ensure_ascii=False,indent=2),encoding="utf-8")
    imp=pd.DataFrame({"feature":FEATURES,"importance":model.get_feature_importance()}).sort_values("importance",ascending=False); imp["rank"]=range(1,len(imp)+1); imp.to_csv(out/"feature_importance.csv",index=False,encoding="utf-8-sig"); imp.head(20).to_csv(out/"feature_importance_top20.csv",index=False,encoding="utf-8-sig")
    err=rec[rec.label.ne(rec.predicted_label)].copy(); err.to_csv(out/"misclassified_records.csv",index=False,encoding="utf-8-sig")
    directions=[("正常","联轴器不对中","normal_to_misalignment"),("松动","轴承故障","looseness_to_bearing"),("轴承故障","松动","bearing_to_looseness")]
    for actual,predicted,name in directions:
        pred[(pred.label==actual)&(pred.predicted_label==predicted)].to_csv(out/f"{name}_windows.csv",index=False,encoding="utf-8-sig")
        rec[(rec.label==actual)&(rec.predicted_label==predicted)].to_csv(out/f"{name}_records.csv",index=False,encoding="utf-8-sig")
    window_errors=_error_source_summary(pred,"error_windows","total_windows")
    record_errors=_error_source_summary(rec,"error_records","total_records")
    window_errors.to_csv(out/"window_error_source_summary.csv",index=False,encoding="utf-8-sig")
    record_errors.to_csv(out/"record_error_source_summary.csv",index=False,encoding="utf-8-sig")
    meta={"mode":mode,"channel":channel,"iterations":int(iterations),"params":params,"common":COMMON,"features":FEATURES,"classes":classes,"train_windows":len(train),"test_windows":len(test),"train_records":train.record_id.nunique(),"test_records":test.record_id.nunique()}
    (out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return wm,rm
