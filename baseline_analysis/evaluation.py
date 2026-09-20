import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score

META=["motor","rpm","condition","state","severity"]
def fuse_records(windows,classes):
    p=windows[classes].to_numpy(float); valid=np.isfinite(p).all(1)&(p>=0).all(1)&(p<=1).all(1)&np.isclose(p.sum(1),1,atol=1e-8)
    d=windows.copy(); d["_valid"]=valid; d["_max"]=np.where(valid,p.max(1),np.nan)
    rows=[]
    for rid,g in d.groupby("record_id",sort=True):
        vg=g[g._valid]; probs=vg[classes].mean(); row={"record_id":rid,"label":g.label.iloc[0],"window_count":len(g),"valid_window_count":len(vg),"mean_max_class_probability":vg._max.mean(),**{m:g[m].iloc[0] for m in META},**probs.to_dict()}; row["predicted_label"]=probs.idxmax(); rows.append(row)
    return pd.DataFrame(rows)

def evaluate_predictions(y_true,y_pred,labels):
    report=classification_report(y_true,y_pred,labels=labels,output_dict=True,zero_division=0)
    normal=np.asarray(y_true)=="正常"; pred=np.asarray(y_pred)
    return {"accuracy":accuracy_score(y_true,y_pred),"macro_f1":f1_score(y_true,y_pred,labels=labels,average="macro",zero_division=0),"weighted_f1":f1_score(y_true,y_pred,labels=labels,average="weighted",zero_division=0),"normal_false_positive_rate":float(np.mean(pred[normal]!="正常")) if normal.any() else None,"classification_report":report,"confusion_matrix":confusion_matrix(y_true,y_pred,labels=labels).tolist()}
