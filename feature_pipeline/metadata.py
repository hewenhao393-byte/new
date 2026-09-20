from pathlib import Path
import re
import pandas as pd

ALLOWED_LABELS={"正常","转子不平衡","联轴器不对中","松动","轴承故障","汽蚀"}
FAILURE_MAP={"Healthy":"正常状态","Cavitation suction":"吸入口汽蚀","Cavitation discharge":"出口汽蚀","Bearing BPFI":"轴承内圈故障","Bearing BPFO":"轴承外圈故障","Bearing BSF":"轴承滚动体故障","Bearing contaminated":"轴承污染","Bearing pump":"泵轴承故障","Pump bearing":"泵轴承故障","Unbalance pump":"泵不平衡","Unbalance motor":"电机不平衡","Pump unbalance":"泵不平衡","Motor unbalance":"电机不平衡","Align angular":"角向不对中","Align parallel":"平行不对中","Align combination":"组合不对中","Soft foot":"软脚","Loose foot motor":"电机地脚松动","Loose foot pump":"泵地脚松动"}

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

def load_measurement_rows(path):
    frame=pd.read_excel(path,sheet_name="Ordered Measurements",header=1)
    rows=[]
    for r in frame.dropna(subset=["Setup","Failure description","Speed (%)","Speed (RPM) +- 5"]).to_dict("records"):
        digits=re.findall(r"\d+",str(r["Setup"]));
        if not digits: continue
        try: speed=float(r["Speed (%)"])
        except (TypeError,ValueError): continue
        speed=int(round(speed*100 if speed<=1 else speed))
        severity=r.get("Severity"); missing=pd.isna(severity) or str(severity).strip() in {"","N/A"}
        description=str(r["Failure description"]).strip(); found=re.search(r"(\d+)$",description)
        description_base=description[:found.start()].strip() if found else description
        base=FAILURE_MAP.get(description_base,description_base)
        if found: suffix=found.group(1)
        elif base=="正常状态" and missing: suffix="1"
        elif missing: suffix=""
        else:
            try: suffix=str(int(float(severity)))
            except ValueError: suffix=str(severity).strip()
        try: rpm=float(r["Speed (RPM) +- 5"])
        except (TypeError,ValueError): continue
        rows.append({"device_id":f"Motor-{int(digits[0])}","speed_percent":speed,"raw_fault":base+suffix,"severity":None if missing else severity,"rpm":rpm})
    return rows

def merge_measurement_metadata(records, rows):
    out=records.copy(); audits=[]; severities=[]; rpms=[]
    for _,r in out.iterrows():
        same_speed=[x for x in rows if str(x.get("device_id")).lower()==str(r.device_id).lower() and int(x.get("speed_percent",-1))==int(r.speed_percent)]
        matches=[x for x in same_speed if x.get("raw_fault")==r.raw_fault]
        consistent=len(matches)>0 and len({(x.get("rpm"),str(x.get("severity"))) for x in matches})==1
        status="matched" if len(matches)==1 else ("matched_consistent" if consistent else ("ambiguous" if len(matches)>1 else "unmatched"))
        m=matches[0] if len(matches)==1 or consistent else {}
        if not m and same_speed:
            counts=pd.Series([x["rpm"] for x in same_speed]).value_counts(); m={"rpm":float(counts.index[0])}
        severities.append(m.get("severity")); rpms.append(m.get("rpm")); audits.append({"group_id":r.group_id,"status":status})
    out["severity"]=severities
    out["rpm"]=rpms
    out["state"]=out["raw_fault"]
    out["condition"]=out["device_id"].astype(str)+"/"+out["speed_percent"].astype(str)
    return out,pd.DataFrame(audits)
