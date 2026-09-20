from pathlib import Path
import numpy as np
import pandas as pd
from .config import FEATURE_NAMES

KEYS=["record_id","split","window_id","start_sample","end_sample"]
WAVE=[f"wp_energy_ratio_{i}" for i in range(8)]
BANDS=["band_energy_5_300_ratio","band_energy_300_1000_ratio","band_energy_1000_3000_ratio","band_energy_3000_5000_ratio"]

def _result(check,measured,threshold,passed,detail=""):
    return {"check":check,"measured":measured,"threshold":threshold,"passed":bool(passed),"detail":detail}

def accept_feature_tables(source_root):
    root=Path(source_root); rows=[]; tables={}
    for mode,folder in [("record","file_split"),("temporal","temporal_split")]:
        tables[mode]={ch:pd.read_csv(root/folder/f"features_ch{ch}.csv",low_memory=False) for ch in (3,4,5)}
        for ch,d in tables[mode].items():
            actual=[c for c in d.columns if c in FEATURE_NAMES]
            rows.append(_result(f"{mode}_ch{ch}_feature_contract",len(actual),43,actual==list(FEATURE_NAMES)))
            a=d[list(FEATURE_NAMES)].to_numpy(float); rows.append(_result(f"{mode}_ch{ch}_nonfinite",int((~np.isfinite(a)).sum()),0,np.isfinite(a).all()))
            werr=float(np.max(np.abs(d[WAVE].sum(axis=1)-1))); berr=float(np.max(np.abs(d[BANDS].sum(axis=1)-1)))
            rows.append(_result(f"{mode}_ch{ch}_wavelet_sum_max_error",werr,1e-8,werr<1e-8)); rows.append(_result(f"{mode}_ch{ch}_band_sum_max_error",berr,1e-6,berr<1e-6))
        base=tables[mode][3]
        for ch in (4,5):
            aligned=base[KEYS].equals(tables[mode][ch][KEYS]); labels=base.label.equals(tables[mode][ch].label); rpm=np.allclose(base.rpm,tables[mode][ch].rpm,equal_nan=True)
            rows += [_result(f"{mode}_ch3_ch{ch}_key_alignment",int(not aligned),0,aligned),_result(f"{mode}_ch3_ch{ch}_label_alignment",int(not labels),0,labels),_result(f"{mode}_ch3_ch{ch}_rpm_alignment",int(not rpm),0,rpm)]
    r=tables["record"][3]; overlap=set(r.loc[r.split.eq("train_dev"),"record_id"])&set(r.loc[r.split.eq("test"),"record_id"]); rows.append(_result("record_train_test_overlap",len(overlap),0,not overlap))
    d=tables["temporal"][3]; bounds=pd.read_csv(root/"temporal_split/temporal_split.csv").set_index("record_id")
    bad_guard=0; bad_block=0
    for rec,g in d.groupby("record_id",sort=False):
        b=bounds.loc[rec]; guard=((g.start_sample<b.guard_end)&(g.end_sample>b.guard_start)); bad_guard+=int(guard.sum())
        tr=g[g.split.eq("train")]; rel_start=tr.start_sample-b.train_start; rel_end=tr.end_sample-1-b.train_start; bad_block+=int(((rel_start//19200)!=(rel_end//19200)).sum())
    rows.append(_result("temporal_guard_crossing_windows",bad_guard,0,bad_guard==0)); rows.append(_result("temporal_train_block_crossing_windows",bad_block,0,bad_block==0)); rows.append(_result("temporal_guard_length_error",int(((bounds.guard_end-bounds.guard_start)!=12000).sum()),0,((bounds.guard_end-bounds.guard_start)==12000).all()))
    report=pd.DataFrame(rows)
    if not report.passed.all(): raise ValueError("feature table acceptance failed: "+", ".join(report.loc[~report.passed,"check"]))
    return report,tables

def write_acceptance(report,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False); report.to_csv(out/"acceptance_checks.csv",index=False,encoding="utf-8-sig")
    columns=list(report.columns); table=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    table += ["| "+" | ".join(str(row[c]) for c in columns)+" |" for _,row in report.iterrows()]
    lines=["# 六张特征表验收报告","",f"- 检查项：{len(report)}",f"- 通过：{int(report.passed.sum())}",f"- 失败：{int((~report.passed).sum())}","",*table]
    (out/"acceptance_report.md").write_text("\n".join(lines),encoding="utf-8")
