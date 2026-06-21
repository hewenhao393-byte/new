from __future__ import annotations

import numpy as np
import pandas as pd

from config import PipelineConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.pipeline import METADATA_COLUMNS


def validate_feature_csv(path, config: PipelineConfig) -> dict[str, object]:
    frame = pd.read_csv(path)
    expected_columns = METADATA_COLUMNS + FEATURE_COLUMNS
    if list(frame.columns) != expected_columns:
        raise ValueError("特征表字段顺序不正确")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("特征表存在NaN或Inf")
    wavelet_sum = frame[[f"wp_energy_{name}" for name in ("aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd")]].sum(axis=1)
    if not np.allclose(wavelet_sum.to_numpy(), 1.0, atol=config.wavelet_ratio_tolerance):
        raise ValueError("小波包能量占比之和不为1")
    return {
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "labels": frame["label"].value_counts().to_dict(),
    }
