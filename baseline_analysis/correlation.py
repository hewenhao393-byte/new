import pandas as pd
from .config import PROTECTED_FEATURES

def analyze_correlations(df,features,training_split):
    train=df[df.split.eq(training_split)][features]
    pearson=train.corr(method="pearson"); spearman=train.corr(method="spearman")
    rows=[]
    for i,a in enumerate(features):
        for b in features[i+1:]:
            r=pearson.loc[a,b]
            if abs(r)>=.95:
                protected=a in PROTECTED_FEATURES or b in PROTECTED_FEATURES
                rows.append({"feature_a":a,"feature_b":b,"pearson_r":r,"abs_pearson_r":abs(r),"review_action":"manual_review_physical_feature" if protected else "candidate_remove_one"})
    return {"pearson":pearson,"spearman":spearman,"pearson_pairs":pd.DataFrame(rows)}
