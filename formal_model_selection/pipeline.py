import hashlib,json,os,uuid
from pathlib import Path
import pandas as pd
from baseline_analysis.acceptance import accept_feature_tables,write_acceptance
from .config import CANDIDATES,CHECKPOINTS,COMMON,FEATURES,LABEL_ORDER
from .cv import run_candidate_cv
from .modeling import train_final
from .reporting import create_reports,save_run_confusions
from .selection import choose_unified,make_shared_folds,summarize_scores
from .verification import verify_run

def _sha(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _recalls(m,prefix): return {f"{prefix}_recall_{x}":m["classification_report"][x]["recall"] for x in LABEL_ORDER}

def run_pipeline(source_root,output_root):
    source=Path(source_root); final=Path(output_root)
    if final.exists(): raise FileExistsError(final)
    files=[source/folder/f"features_ch{ch}.csv" for folder in ("file_split","temporal_split") for ch in (3,4,5)]
    hashes={str(p.resolve()):_sha(p) for p in files}
    acceptance,tables=accept_feature_tables(source)
    train3=tables["record"][3].query("split=='train_dev'"); folds,fold_hash=make_shared_folds(train3)
    for ch in (4,5):
        if set(tables["record"][ch].query("split=='train_dev'").record_id)!=set(train3.record_id): raise ValueError("channel record mismatch")
    fold_scores=run_candidate_cv(tables["record"],folds,fold_hash); summary=summarize_scores(fold_scores); chosen=choose_unified(summary)
    stage=final.parent/(final.name+".staging-"+uuid.uuid4().hex); stage.mkdir(parents=True)
    try:
        write_acceptance(acceptance,stage/"feature_acceptance"); folds.to_csv(stage/"shared_record_folds.csv",index=False,encoding="utf-8-sig"); fold_scores.to_csv(stage/"cv_fold_scores.csv",index=False,encoding="utf-8-sig"); summary.to_csv(stage/"cv_candidate_summary.csv",index=False,encoding="utf-8-sig")
        params=CANDIDATES[chosen["candidate"]]; rows=[]
        for mode in ("record","temporal"):
            for ch in (3,4,5):
                out=stage/"models"/mode/f"ch{ch}"; out.parent.mkdir(parents=True,exist_ok=True); wm,rm=train_final(tables[mode][ch],mode,ch,params,int(chosen["iteration"]),out); save_run_confusions(out,wm,rm,f"CH{ch} {mode}")
                row={"channel":f"CH{ch}","split_mode":mode,"window_accuracy":wm["accuracy"],"window_macro_f1":wm["macro_f1"],"window_weighted_f1":wm["weighted_f1"],"record_accuracy":rm["accuracy"],"record_macro_f1":rm["macro_f1"],"record_weighted_f1":rm["weighted_f1"],**_recalls(wm,"window"),**_recalls(rm,"record")}; rows.append(row)
        comparison=pd.DataFrame(rows); comparison.to_csv(stage/"final_model_comparison.csv",index=False,encoding="utf-8-sig"); create_reports(stage,fold_scores,summary,chosen,comparison)
        manifest={"source":str(source.resolve()),"input_sha256":hashes,"features":FEATURES,"labels":LABEL_ORDER,"candidates":CANDIDATES,"checkpoints":CHECKPOINTS,"cv_splits":3,"fold_hash":fold_hash,"score":"equal mean of CH3/CH4/CH5 fold means","tolerance":.002,"selected":{"candidate":chosen["candidate"],"iteration":int(chosen["iteration"]),"params":params},"common":COMMON}
        (stage/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        verify_run(source,stage)
        if hashes!={str(p.resolve()):_sha(p) for p in files}: raise RuntimeError("input changed during run")
        if final.exists(): raise FileExistsError(final)
        os.rename(stage,final)
    except Exception as e:
        raise RuntimeError(f"run failed; staging retained at {stage}: {e}") from e
    return comparison
