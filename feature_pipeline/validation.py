import json
from pathlib import Path
import numpy as np
from .config import FEATURE_NAMES
from .features import BAND_RATIO_NAMES

class ValidationError(ValueError): pass

def validate_channel_alignment(tables):
    keys=["record_id","split","window_id","start_sample","end_sample"]
    base=tables[3][keys].reset_index(drop=True)
    for ch in (4,5):
        if not base.equals(tables[ch][keys].reset_index(drop=True)): raise ValidationError("channel alignment mismatch")

def validate_table(df):
    if not set(FEATURE_NAMES).issubset(df): raise ValidationError("missing feature columns")
    a=df[list(FEATURE_NAMES)].to_numpy(float)
    if not np.isfinite(a).all(): raise ValidationError("non-finite feature")
    if df.duplicated(["record_id","split","window_id"]).any(): raise ValidationError("duplicate window")
    bands=df[list(BAND_RATIO_NAMES)].sum(axis=1).to_numpy()
    if np.max(np.abs(bands-1))>=1e-6: raise ValidationError("band ratio sum")
    wave=df[[f"wp_energy_ratio_{i}" for i in range(8)]].sum(axis=1).to_numpy()
    if np.max(np.abs(wave-1))>=1e-8: raise ValidationError("wavelet ratio sum")

def verify_output(root):
    import pandas as pd
    report={"passed":True,"modes":{}}
    for folder in ("file_split","temporal_split"):
        tables={ch:pd.read_csv(Path(root)/folder/f"features_ch{ch}.csv") for ch in (3,4,5)}
        for df in tables.values(): validate_table(df)
        validate_channel_alignment(tables)
        if folder=="file_split":
            d=pd.read_csv(Path(root)/folder/"file_split.csv")
            if d.groupby("record_id").split.nunique().max()!=1: raise ValidationError("record leakage")
        else:
            d=pd.read_csv(Path(root)/folder/"temporal_split.csv")
            if not ((d.guard_end-d.guard_start)==12000).all(): raise ValidationError("guard is not 1.0 second")
            if not (((d.train_end<=d.guard_start)&(d.guard_end<=d.test_start))|((d.test_end<=d.guard_start)&(d.guard_end<=d.train_start))).all(): raise ValidationError("temporal ranges overlap")
        report["modes"][folder]={"rows_per_channel":len(tables[3])}
    return report

def write_audit(path,report):
    Path(path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
