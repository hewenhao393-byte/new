import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .config import CANDIDATES,CHECKPOINTS,CV_SPLITS,SEED

def make_shared_folds(train):
    rec=train[["record_id","label"]].drop_duplicates()
    if rec.groupby("record_id").label.nunique().max()!=1: raise ValueError("record label mismatch")
    splitter=StratifiedGroupKFold(n_splits=CV_SPLITS,shuffle=True,random_state=SEED)
    rows=[]
    for fold,(fit,val) in enumerate(splitter.split(rec,rec.label,groups=rec.record_id),1):
        val_ids=set(rec.iloc[val].record_id)
        for rid in rec.record_id: rows.append({"fold":fold,"record_id":rid,"role":"validation" if rid in val_ids else "fit"})
    out=pd.DataFrame(rows).sort_values(["fold","role","record_id"]).reset_index(drop=True)
    raw=out.to_csv(index=False).encode(); return out,hashlib.sha256(raw).hexdigest()

def summarize_scores(fold_scores):
    g=(fold_scores.groupby(["candidate","iteration","channel"],as_index=False)
       .agg(mean=("record_macro_f1","mean"),std=("record_macro_f1","std")))
    means=g.pivot(index=["candidate","iteration"],columns="channel",values="mean").reset_index()
    stds=g.pivot(index=["candidate","iteration"],columns="channel",values="std").reset_index()
    rows=[]
    for _,m in means.iterrows():
        s=stds[(stds.candidate==m.candidate)&(stds.iteration==m.iteration)].iloc[0]
        rows.append({"candidate":m.candidate,"iteration":int(m.iteration),"ch3_mean":m[3],"ch4_mean":m[4],"ch5_mean":m[5],"ch3_std":s[3],"ch4_std":s[4],"ch5_std":s[5],"score":np.mean([m[3],m[4],m[5]]),"stability":np.mean([s[3],s[4],s[5]])})
    return pd.DataFrame(rows).sort_values(["candidate","iteration"]).reset_index(drop=True)

def choose_unified(summary,tolerance=.002):
    if summary.empty or not np.isfinite(summary[["score","stability"]]).all().all(): raise ValueError("invalid selection summary")
    best=summary.score.max()
    # The specification treats a score gap of exactly 0.002 as inside the
    # simplified-candidate band.  Allow only a tiny numerical guard so binary
    # floating-point representation cannot exclude that inclusive boundary.
    near=summary[summary.score>=best-tolerance-1e-12].copy()
    near["depth"]=near.candidate.map(lambda x:CANDIDATES[x]["depth"])
    near["l2"]=near.candidate.map(lambda x:CANDIDATES[x]["l2_leaf_reg"])
    near=near.sort_values(["depth","iteration","l2","stability","candidate"],ascending=[True,True,False,True,True])
    return near.iloc[0].to_dict()
