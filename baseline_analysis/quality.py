import numpy as np
import pandas as pd

def describe_features(df,features,group_cols):
    rows=[]
    grouped=[((),df)] if not group_cols else df.groupby(group_cols,dropna=False,sort=True)
    for keys,g in grouped:
        if not isinstance(keys,tuple): keys=(keys,)
        base=dict(zip(group_cols,keys))
        for f in features:
            s=g[f].astype(float)
            rows.append({**base,"feature":f,"count":s.count(),"mean":s.mean(),"std":s.std(),"median":s.median(),"q1":s.quantile(.25),"q3":s.quantile(.75),"iqr":s.quantile(.75)-s.quantile(.25)})
    return pd.DataFrame(rows)

def pairwise_effects(df,features,column,a,b):
    rows=[]
    for f in features:
        x=df.loc[df[column].eq(a),f].astype(float); y=df.loc[df[column].eq(b),f].astype(float)
        pooled=np.sqrt(((len(x)-1)*x.var()+(len(y)-1)*y.var())/max(1,len(x)+len(y)-2))
        iq=(pd.concat([x,y]).quantile(.75)-pd.concat([x,y]).quantile(.25))
        rows.append({"feature":f,"group_a":a,"group_b":b,"mean_difference":x.mean()-y.mean(),"standardized_mean_difference":(x.mean()-y.mean())/(pooled+1e-12),"median_difference":x.median()-y.median(),"robust_median_difference":(x.median()-y.median())/(iq+1e-12)})
    return pd.DataFrame(rows)

def record_variability(df,features):
    rec=df.groupby("record_id",sort=True)[features].mean()
    rows=[]
    for f in features:
        s=rec[f]; rows.append({"feature":f,"record_count":len(s),"record_mean":s.mean(),"record_std":s.std(),"record_median":s.median(),"record_q1":s.quantile(.25),"record_q3":s.quantile(.75),"record_iqr":s.quantile(.75)-s.quantile(.25),"record_cv":s.std()/(abs(s.mean())+1e-12)})
    return pd.DataFrame(rows)
