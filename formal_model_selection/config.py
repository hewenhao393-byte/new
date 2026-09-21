from baseline_analysis.config import FEATURE_COLUMNS, LABEL_ORDER

FEATURES=list(FEATURE_COLUMNS)
CHECKPOINTS=[50,100,150,200,300,400,500]
MAX_ITERATIONS=500
CV_SPLITS=3
SEED=2026
COMMON={"loss_function":"MultiClass","auto_class_weights":"SqrtBalanced","random_seed":SEED,"thread_count":4,"allow_writing_files":False,"verbose":False}
CANDIDATES={
    "P1":{"depth":8,"learning_rate":.05,"l2_leaf_reg":100,"random_strength":5,"rsm":.7},
    "P2":{"depth":6,"learning_rate":.05,"l2_leaf_reg":100,"random_strength":5,"rsm":.7},
    "P5":{"depth":8,"learning_rate":.05,"l2_leaf_reg":200,"random_strength":5,"rsm":.7},
}
META=["record_id","window_id","label","motor","rpm","condition","state","severity","start_sample","end_sample"]
