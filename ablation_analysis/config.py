from baseline_analysis.config import FEATURE_COLUMNS, LABEL_ORDER, MODEL_PARAMS


FEATURE_43 = list(FEATURE_COLUMNS)
REMOVED_FEATURES = ["std", "band_energy_3000_5000_ratio", "wp_energy_ratio_7"]
FEATURE_40 = [feature for feature in FEATURE_43 if feature not in REMOVED_FEATURES]

ITERATION_GRID = list(range(20, 801, 20))
MAX_ITERATIONS = 800
CV_SPLITS = 5
JOIN_KEYS = ["record_id", "window_id", "start_sample", "end_sample"]

if len(FEATURE_43) != 43 or len(FEATURE_40) != 40:
    raise ValueError("ablation feature contract must contain exactly 43 full and 40 reduced features")

