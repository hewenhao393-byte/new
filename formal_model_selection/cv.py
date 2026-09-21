import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from baseline_analysis.evaluation import evaluate_predictions,fuse_records
from .config import CANDIDATES,CHECKPOINTS,COMMON,FEATURES,LABEL_ORDER,MAX_ITERATIONS,META

def _pred_frame(frame,prob,classes):
    missing=[column for column in META if column not in frame.columns]
    if missing:
        raise ValueError(f"missing prediction metadata: {missing}")
    prob=np.asarray(prob,dtype=float)
    if prob.shape!=(len(frame),len(classes)):
        raise ValueError("probability shape does not match rows/classes")
    valid=(np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
           and np.allclose(prob.sum(axis=1),1.0,atol=1e-8))
    if not valid:
        raise ValueError("invalid probability matrix")
    out=frame[META].copy(); out["predicted_label"]=[classes[i] for i in prob.argmax(1)]
    for i,c in enumerate(classes): out[c]=prob[:,i]
    return out

def run_candidate_cv(tables,fold_manifest,fold_hash):
    rows=[]
    for candidate,params in CANDIDATES.items():
        for channel in (3,4,5):
            data=tables[channel]; train=data[data.split.eq("train_dev")].reset_index(drop=True)
            for fold in sorted(fold_manifest.fold.unique()):
                val_ids=set(fold_manifest.query("fold==@fold and role=='validation'").record_id)
                val_mask=train.record_id.isin(val_ids); fit=train[~val_mask]; val=train[val_mask]
                if set(fit.record_id)&set(val.record_id): raise AssertionError("record leakage")
                model=CatBoostClassifier(**COMMON,**params,iterations=MAX_ITERATIONS)
                model.fit(Pool(fit[FEATURES],fit.label)); classes=list(model.classes_)
                found=[]
                for iteration,prob in enumerate(model.staged_predict_proba(val[FEATURES]),1):
                    if iteration not in CHECKPOINTS: continue
                    found.append(iteration); pred=_pred_frame(val,np.asarray(prob),classes); rec=fuse_records(pred,classes)
                    metric=evaluate_predictions(rec.label,rec.predicted_label,LABEL_ORDER)
                    rows.append({"candidate":candidate,"channel":channel,"fold":fold,"iteration":iteration,"record_macro_f1":metric["macro_f1"],"validation_records":len(rec),"fold_hash":fold_hash})
                if found!=CHECKPOINTS: raise AssertionError("incomplete checkpoints")
    return pd.DataFrame(rows)
