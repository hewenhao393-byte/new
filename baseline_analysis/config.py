from feature_pipeline.config import FEATURE_NAMES

FEATURE_COLUMNS=list(FEATURE_NAMES)

LABEL_ORDER=["正常","转子不平衡","联轴器不对中","松动","轴承故障","汽蚀"]
FIXED_PARAMS={"loss_function":"MultiClass","iterations":540,"depth":8,"learning_rate":.05,"l2_leaf_reg":100,"random_strength":5,"rsm":.7,"auto_class_weights":"SqrtBalanced","random_seed":2026,"thread_count":4,"allow_writing_files":False,"verbose":False,"parameter_role":"fixed baseline; not claimed optimal"}
MODEL_PARAMS={k:v for k,v in FIXED_PARAMS.items() if k!="parameter_role"}
PROTECTED_FEATURES={"rot_2x_1x_ratio","rot_3x_1x_ratio","rot_05x_1x_ratio","harmonic_energy_ratio_1x_5x","harmonic_energy_ratio_3x_5x","rot_2x_harmonic_ratio","noninteger_harmonic_energy_ratio"}
