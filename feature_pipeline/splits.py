import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CONFIG

def make_record_split(records):
    d=records.copy().sort_values("record_id").reset_index(drop=True)
    parts=[d[c].astype(str) for c in ("label","device_id","speed_percent","raw_fault") if c in d]
    strat=parts[0]
    for part in parts[1:]: strat=strat+"|"+part
    if strat.value_counts().min()<2: strat=d["label"]
    train,test=train_test_split(d.index,test_size=.2,random_state=CONFIG.random_seed,stratify=strat)
    d["split"]="test"; d.loc[train,"split"]="train_dev"
    return d

def temporal_bounds(record_id,n_samples,seed=2026,force_direction=None):
    if n_samples!=144_000: raise ValueError("temporal mode requires exactly 144000 samples")
    if force_direction is None:
        bit=int(hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest(),16)&1
        force_direction="test_tail" if bit else "test_head"
    if force_direction=="test_tail": return {"train":(0,96_000),"guard":(96_000,108_000),"test":(108_000,144_000)}
    if force_direction=="test_head": return {"test":(0,36_000),"guard":(36_000,48_000),"train":(48_000,144_000)}
    raise ValueError("invalid split direction")
