import numpy as np
import pandas as pd
import pytest

from baseline_analysis.config import FEATURE_COLUMNS, FEATURE_NAMES, FIXED_PARAMS, LABEL_ORDER
from baseline_analysis.quality import class_separation, describe_features, pairwise_effects, record_variability
from baseline_analysis.correlation import analyze_correlations
from baseline_analysis.evaluation import evaluate_predictions, fuse_records
from baseline_analysis.acceptance import write_acceptance
from baseline_analysis.pipeline import _recalls

def test_baseline_contract():
    assert len(FEATURE_NAMES)==43
    assert isinstance(FEATURE_COLUMNS,list)
    assert FIXED_PARAMS["iterations"]==540
    assert "fixed baseline" in FIXED_PARAMS["parameter_role"]
    assert "StandardScaler" not in FIXED_PARAMS

def test_quality_statistics_and_record_first_variability():
    d=pd.DataFrame({"record_id":["a","a","b","b"],"label":["x"]*2+["y"]*2,"f":[1.,3.,5.,7.]})
    s=describe_features(d,["f"],["label"])
    assert set(["mean","std","median","q1","q3","iqr"]).issubset(s)
    e=pairwise_effects(d,["f"],"label","x","y")
    assert e.loc[0,"mean_difference"]==-4
    v=record_variability(d,["f"])
    assert v.loc[0,"record_count"]==2
    c=class_separation(d,["f"])
    assert c.loc[0,"eta_squared"]>0

def test_pearson_controls_pairs_and_protected_features_are_manual_review():
    x=np.arange(20,dtype=float)
    d=pd.DataFrame({"split":["train"]*20,"rms":x,"std":2*x,"rot_2x_1x_ratio":3*x})
    result=analyze_correlations(d,["rms","std","rot_2x_1x_ratio"],"train")
    assert len(result["pearson_pairs"])==3
    protected=result["pearson_pairs"].query("feature_a=='rot_2x_1x_ratio' or feature_b=='rot_2x_1x_ratio'")
    assert protected.review_action.eq("manual_review_physical_feature").all()

def test_record_fusion_counts_and_metrics():
    rows=[]
    for rid,true,probs in [("a","正常",[[.8,.2],[.6,.4]]),("b","松动",[[.2,.8]])]:
        for p in probs: rows.append({"record_id":rid,"label":true,"正常":p[0],"松动":p[1],"motor":"M","rpm":1,"condition":"c","state":"s","severity":np.nan})
    fused=fuse_records(pd.DataFrame(rows),["正常","松动"])
    assert fused.set_index("record_id").loc["a","window_count"]==2
    assert fused.valid_window_count.eq(fused.window_count).all()
    assert fused.mean_max_class_probability.notna().all()
    metrics=evaluate_predictions(fused.label,fused.predicted_label,["正常","松动"])
    assert metrics["accuracy"]==1
    assert metrics["normal_false_positive_rate"]==0

def test_acceptance_report_has_no_optional_tabulate_dependency(tmp_path):
    report=pd.DataFrame([{"check":"x","measured":0,"threshold":0,"passed":True,"detail":""}])
    write_acceptance(report,tmp_path/"acceptance")
    assert "| check |" in (tmp_path/"acceptance/acceptance_report.md").read_text()

def test_comparison_recall_columns_are_level_specific():
    metrics={"classification_report":{label:{"recall":i/10} for i,label in enumerate(LABEL_ORDER)}}
    recalls=_recalls(metrics)
    assert recalls["召回_正常"] if False else recalls["recall_正常"]==0
    assert len(recalls)==6
