from pathlib import Path
import pandas as pd

ALLOWED_LABELS={"正常","转子不平衡","联轴器不对中","松动","轴承故障","汽蚀"}

def make_record_id(device_id, speed_percent, state, column):
    return f"{device_id}::{int(speed_percent)}::{state}::{column}"

def load_record_manifest(path):
    data=pd.read_csv(path,dtype={"record_column":str})
    data=data[data["eligible"].astype(bool)].copy()
    if not data.group_id.is_unique or not set(data.label).issubset(ALLOWED_LABELS):
        raise ValueError("invalid record manifest")
    for col in ("ch3_path","ch4_path","ch5_path"):
        if not data[col].map(lambda p: Path(p).is_file()).all(): raise ValueError(f"missing {col}")
    data["record_id"]=data.group_id
    return data

def merge_measurement_metadata(records, rows):
    out=records.copy(); audits=[]; severities=[]; rpms=[]
    for _,r in out.iterrows():
        matches=[x for x in rows if str(x.get("device_id")).lower()==str(r.device_id).lower() and int(x.get("speed_percent",-1))==int(r.speed_percent) and x.get("raw_fault")==r.raw_fault]
        status="matched" if len(matches)==1 else ("ambiguous" if len(matches)>1 else "unmatched")
        m=matches[0] if len(matches)==1 else {}
        severities.append(m.get("severity")); rpms.append(m.get("rpm")); audits.append({"group_id":r.group_id,"status":status})
    out["severity"]=severities
    if "rpm" not in out: out["rpm"]=rpms
    return out,pd.DataFrame(audits)
