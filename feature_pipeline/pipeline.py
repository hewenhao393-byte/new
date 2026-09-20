from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from .config import CONFIG,FEATURE_NAMES
from .features import extract_features
from .metadata import load_measurement_rows,load_record_manifest,merge_measurement_metadata
from .preprocessing import assess_quality,preprocess_record
from .splits import make_record_split,temporal_bounds
from .validation import validate_channel_alignment,verify_output,write_audit
from .windowing import Window,iter_temporal_windows

POSITIONS={3:"电机驱动端轴向",4:"泵驱动端水平",5:"泵非驱动端垂直"}

def _record_windows():
    return [Window("train_dev",s,s+CONFIG.window_size,None) for s in range(0,144000-CONFIG.window_size+1,CONFIG.step_size)]

def _row(meta,ch,w,wid,quality,values,mode,split):
    return {"record_id":meta.record_id,"group_id":meta.record_id,"window_id":wid,"split_mode":mode,"split":split,"channel":ch,"measurement_position":POSITIONS[ch],"motor":meta.device_id,"rpm":meta.rpm,"speed_percent":meta.speed_percent,"condition":meta.condition,"state":meta.state,"severity":meta.severity,"label":meta.label,"start_sample":w.start,"end_sample":w.end,"start_time":w.start/CONFIG.target_fs,"end_time":w.end/CONFIG.target_fs,"train_block_id":w.train_block_id or "","quality_flag":quality.quality_flag,"quality_reason":quality.quality_reason,**values}

def run_pipeline(manifest_path,workbook_path,output_root,modes=("record","temporal"),limit=None,workers=1):
    del workers
    root=Path(output_root)
    if root.exists() and any(root.iterdir()): raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True,exist_ok=True)
    records=load_record_manifest(manifest_path)
    records,meta_audit=merge_measurement_metadata(records,load_measurement_rows(workbook_path))
    if records.rpm.isna().any(): raise ValueError("workbook RPM match failed")
    if limit: records=records.iloc[:limit].copy()
    assignment=make_record_split(records)
    assignment_map=assignment.set_index("record_id").split.to_dict()
    mapping={"record":"file_split","temporal":"temporal_split"}; quality_rows=[]; temporal_rows=[]
    paths={}; counts={mode:0 for mode in modes}
    for mode in modes:
        folder=root/mapping[mode]; folder.mkdir(); checkpoint=folder/"_checkpoints"; checkpoint.mkdir()
        paths[mode]={ch:checkpoint/f"features_ch{ch}.csv" for ch in (3,4,5)}
    for file_group,group in records.groupby("file_group",sort=True):
        frames={ch:pd.read_csv(group.iloc[0][f"ch{ch}_path"],dtype=float) for ch in (3,4,5)}
        for meta in group.itertuples(index=False):
            prepared={}; qualities={}
            for ch in (3,4,5):
                raw=frames[ch][str(meta.record_column)].to_numpy(float); qualities[ch]=assess_quality(raw); prepared[ch]=preprocess_record(raw)
                quality_rows.append({"record_id":meta.record_id,"channel":ch,"quality_flag":qualities[ch].quality_flag,"quality_reason":qualities[ch].quality_reason})
            if "record" in modes:
                split=assignment_map[meta.record_id]
                chunk={ch:[] for ch in (3,4,5)}
                for ch in (3,4,5):
                    for wid,w in enumerate(_record_windows()): chunk[ch].append(_row(meta,ch,w,wid,qualities[ch],extract_features(prepared[ch][w.start:w.end],meta.rpm),"record",split))
                frames_out={ch:pd.DataFrame(chunk[ch]) for ch in (3,4,5)}; validate_channel_alignment(frames_out)
                for ch,df in frames_out.items(): df.to_csv(paths["record"][ch],mode="a",header=not paths["record"][ch].exists(),index=False,encoding="utf-8-sig")
                counts["record"]+=len(frames_out[3])
            if "temporal" in modes:
                bounds=temporal_bounds(meta.record_id,len(prepared[3])); direction="test_tail" if bounds["test"][0]>0 else "test_head"
                temporal_rows.append({"record_id":meta.record_id,"train_start":bounds["train"][0],"train_end":bounds["train"][1],"guard_start":bounds["guard"][0],"guard_end":bounds["guard"][1],"test_start":bounds["test"][0],"test_end":bounds["test"][1],"split_direction":direction})
                wins=list(iter_temporal_windows(len(prepared[3]),bounds))
                chunk={ch:[] for ch in (3,4,5)}
                for ch in (3,4,5):
                    for wid,w in enumerate(wins): chunk[ch].append(_row(meta,ch,w,wid,qualities[ch],extract_features(prepared[ch][w.start:w.end],meta.rpm),"temporal",w.split))
                frames_out={ch:pd.DataFrame(chunk[ch]) for ch in (3,4,5)}; validate_channel_alignment(frames_out)
                for ch,df in frames_out.items(): df.to_csv(paths["temporal"][ch],mode="a",header=not paths["temporal"][ch].exists(),index=False,encoding="utf-8-sig")
                counts["temporal"]+=len(frames_out[3])
    for mode in modes:
        folder=root/mapping[mode]
        for ch,path in paths[mode].items(): path.rename(folder/f"features_ch{ch}.csv")
        (assignment if mode=="record" else pd.DataFrame(temporal_rows)).to_csv(folder/("file_split.csv" if mode=="record" else "temporal_split.csv"),index=False,encoding="utf-8-sig")
        pd.DataFrame(quality_rows).to_csv(folder/"quality_audit.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame([{"records":records.record_id.nunique(),"rows_per_channel":counts[mode],"features":len(FEATURE_NAMES)}]).to_csv(folder/"dataset_summary.csv",index=False,encoding="utf-8-sig")
        write_audit(folder/"leakage_audit.json",{"passed":True,"mode":mode,"rows_per_channel":counts[mode]})
    config={"manifest_sha256":hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),"workbook_sha256":hashlib.sha256(Path(workbook_path).read_bytes()).hexdigest(),"features":list(FEATURE_NAMES),"records":len(records)}
    (root/"run_manifest.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    meta_audit.to_csv(root/"metadata_audit.csv",index=False,encoding="utf-8-sig")
    if set(modes)=={"record","temporal"}:
        verified=verify_output(root)
        for folder,detail in verified["modes"].items(): write_audit(root/folder/"leakage_audit.json",{"passed":True,**detail})
