import pandas as pd
from formal_model_selection.config import CANDIDATES,CHECKPOINTS,FEATURES
from formal_model_selection.selection import choose_unified,make_shared_folds,summarize_scores

def test_contracts():
    assert list(CANDIDATES)==["P1","P2","P5"]
    assert CHECKPOINTS==[50,100,150,200,300,400,500]
    assert len(FEATURES)==43

def test_shared_folds_and_selection():
    d=pd.DataFrame({"record_id":[f"r{c}_{i}" for c in range(6) for i in range(6)],"label":[str(c) for c in range(6) for i in range(6)]})
    folds,h=make_shared_folds(d); assert len(h)==64
    for f in (1,2,3):
        a=set(folds.query("fold==@f and role=='fit'").record_id); b=set(folds.query("fold==@f and role=='validation'").record_id); assert not a&b
    rows=[]
    for p,score in [("P1",.900),("P2",.899),("P5",.901)]:
        for ch in (3,4,5):
            for fold in (1,2,3): rows.append({"candidate":p,"iteration":100,"channel":ch,"fold":fold,"record_macro_f1":score})
    summary=summarize_scores(pd.DataFrame(rows)); chosen=choose_unified(summary)
    assert chosen["candidate"]=="P2"

def test_same_depth_prefers_fewer_iterations_then_stronger_l2():
    s=pd.DataFrame([
        {"candidate":"P1","iteration":100,"score":.9,"stability":.02},
        {"candidate":"P5","iteration":100,"score":.9,"stability":.02},
        {"candidate":"P5","iteration":150,"score":.901,"stability":.01},
    ])
    assert choose_unified(s)["candidate"]=="P5"
    assert choose_unified(s)["iteration"]==100
