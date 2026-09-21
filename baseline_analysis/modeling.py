import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import StratifiedGroupKFold
from .config import FEATURE_COLUMNS,FEATURE_NAMES,LABEL_ORDER,MODEL_PARAMS,FIXED_PARAMS
from .evaluation import evaluate_predictions,fuse_records

META=["record_id","window_id","label","motor","rpm","condition","state","severity","start_sample","end_sample"]

def _probability_frame(data,probs,classes):
    out=data[META].copy(); out["predicted_label"]=[classes[i] for i in np.argmax(probs,axis=1)]
    for i,c in enumerate(classes): out[c]=probs[:,i]
    return out

def train_baseline(data,mode,channel,output_dir,run_cv=True):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    train_name="train_dev" if mode=="record" else "train"; train=data[data.split.eq(train_name)].reset_index(drop=True); test=data[data.split.eq("test")].reset_index(drop=True)
    groups=train.record_id if mode=="record" else train.record_id.astype(str)+"::"+train.train_block_id.astype(str)
    cv_rows=[]
    if run_cv:
        splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
        for fold,(fit_idx,val_idx) in enumerate(splitter.split(train,train.label,groups),1):
            model=CatBoostClassifier(**MODEL_PARAMS); model.fit(Pool(train.loc[fit_idx,FEATURE_COLUMNS],train.loc[fit_idx,"label"]))
            probs=model.predict_proba(train.loc[val_idx,FEATURE_COLUMNS]); classes=list(model.classes_); pred=_probability_frame(train.loc[val_idx],probs,classes)
            wm=evaluate_predictions(pred.label,pred.predicted_label,LABEL_ORDER); rec=fuse_records(pred,classes); rm=evaluate_predictions(rec.label,rec.predicted_label,LABEL_ORDER)
            cv_rows.append({"fold":fold,"fit_windows":len(fit_idx),"validation_windows":len(val_idx),"validation_records":train.loc[val_idx,"record_id"].nunique(),"window_accuracy":wm["accuracy"],"window_macro_f1":wm["macro_f1"],"record_accuracy":rm["accuracy"],"record_macro_f1":rm["macro_f1"]})
    pd.DataFrame(cv_rows).to_csv(out/"internal_cv.csv",index=False,encoding="utf-8-sig")
    model=CatBoostClassifier(**MODEL_PARAMS); model.fit(Pool(train[FEATURE_COLUMNS],train.label)); model.save_model(str(out/"model.cbm"))
    probs=model.predict_proba(test[FEATURE_COLUMNS]); classes=list(model.classes_); pred=_probability_frame(test,probs,classes); pred.to_csv(out/"window_predictions.csv",index=False,encoding="utf-8-sig")
    restored=CatBoostClassifier(); restored.load_model(str(out/"model.cbm")); np.testing.assert_allclose(probs,restored.predict_proba(test[FEATURE_COLUMNS]),atol=1e-12)
    window_metrics=evaluate_predictions(pred.label,pred.predicted_label,LABEL_ORDER); records=fuse_records(pred,classes); records.to_csv(out/"record_predictions.csv",index=False,encoding="utf-8-sig"); record_metrics=evaluate_predictions(records.label,records.predicted_label,LABEL_ORDER)
    importance=pd.DataFrame({"feature":FEATURE_COLUMNS,"importance":model.get_feature_importance()}).sort_values("importance",ascending=False); importance["rank"]=np.arange(1,len(importance)+1); importance.to_csv(out/"feature_importance.csv",index=False,encoding="utf-8-sig"); importance.head(20).to_csv(out/"feature_importance_top20.csv",index=False,encoding="utf-8-sig")
    errors=records[records.label.ne(records.predicted_label)].copy(); errors.to_csv(out/"misclassified_records.csv",index=False,encoding="utf-8-sig")
    targeted=errors[((errors.label.eq("正常"))&(errors.predicted_label.eq("联轴器不对中")))|((errors.label.eq("松动"))&(errors.predicted_label.eq("轴承故障")))]; targeted.to_csv(out/"targeted_misclassified_records.csv",index=False,encoding="utf-8-sig")
    record_dimensions=[]
    for col in ["motor","rpm","condition","state","severity"]:
        e=errors.groupby(col,dropna=False).size().rename("misclassified_records"); n=records.groupby(col,dropna=False).size().rename("total_records")
        x=pd.concat([e,n],axis=1).fillna(0).reset_index(); x.insert(0,"dimension",col); x=x.rename(columns={col:"value"}); x["misclassification_rate"]=x.misclassified_records/x.total_records; record_dimensions.append(x)
    pd.concat(record_dimensions,ignore_index=True).to_csv(out/"record_error_source_summary.csv",index=False,encoding="utf-8-sig")
    window_errors=pred[pred.label.ne(pred.predicted_label)].copy()
    totals=pred.groupby("record_id").size().rename("total_window_count")
    by_record=(window_errors.groupby(["record_id","label","predicted_label","motor","rpm","condition","state","severity"],dropna=False).size().rename("misclassified_window_count").reset_index().merge(totals,on="record_id"))
    by_record["misclassification_rate"]=by_record.misclassified_window_count/by_record.total_window_count
    by_record.to_csv(out/"window_errors_by_record.csv",index=False,encoding="utf-8-sig")
    dimensions=[]
    for col in ["motor","rpm","condition","state","severity"]:
        e=window_errors.groupby(col,dropna=False).size().rename("misclassified_windows"); n=pred.groupby(col,dropna=False).size().rename("total_windows")
        x=pd.concat([e,n],axis=1).fillna(0).reset_index(); x.insert(0,"dimension",col); x=x.rename(columns={col:"value"}); x["misclassification_rate"]=x.misclassified_windows/x.total_windows; dimensions.append(x)
    pd.concat(dimensions,ignore_index=True).to_csv(out/"window_error_source_summary.csv",index=False,encoding="utf-8-sig")
    window_errors[((window_errors.label.eq("正常"))&(window_errors.predicted_label.eq("联轴器不对中")))|((window_errors.label.eq("松动"))&(window_errors.predicted_label.eq("轴承故障")))].to_csv(out/"targeted_window_errors.csv",index=False,encoding="utf-8-sig")
    def save(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    save("window_metrics.json",window_metrics); save("record_metrics.json",record_metrics); save("metadata.json",{"mode":mode,"channel":channel,"parameters":FIXED_PARAMS,"features":list(FEATURE_NAMES),"classes":classes,"train_windows":len(train),"test_windows":len(test),"train_records":train.record_id.nunique(),"test_records":test.record_id.nunique(),"shap":{"executed":False,"reason":"optional; omitted to keep fixed baseline stage bounded"}})
    return {"window":window_metrics,"record":record_metrics,"importance":importance,"cv":pd.DataFrame(cv_rows)}
