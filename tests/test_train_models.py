from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ModelingConfig
from pump_diagnosis.model_training import _stratified_sample, train_and_evaluate_models


def _make_model_frame(rows_per_class: int = 12) -> pd.DataFrame:
    labels = ["正常", "汽蚀", "轴承故障"]
    rows = []
    for class_index, label in enumerate(labels):
        for sample_index in range(rows_per_class):
            base = class_index * 2.0 + sample_index * 0.03
            rows.append(
                {
                    "label": label,
                    "sample_id": f"{label}_{sample_index}",
                    "run_id": f"{label}_{sample_index}",
                    "file_path": f"/tmp/{label}_{sample_index}.csv",
                    "machine_id": "Motor-4",
                    "condition_id": f"condition_{label}",
                    "rpm": 2070.0,
                    "channel": 4,
                    "window_index": sample_index,
                    "window_start": sample_index * 0.1,
                    "window_end": sample_index * 0.1 + 0.2,
                    "original_fs": 20_000,
                    "processed_fs": 12_000,
                    "feature_a": base,
                    "feature_b": np.sin(base),
                    "feature_c": np.cos(base),
                    "feature_d": base * base,
                }
            )
    return pd.DataFrame(rows)


def test_stratified_sample_preserves_classes_and_limit() -> None:
    frame = _make_model_frame(rows_per_class=10)
    sampled = _stratified_sample(frame, sample_size=12, random_state=3)

    assert len(sampled) == 12
    assert set(sampled["label"]) == {"正常", "汽蚀", "轴承故障"}


def test_train_and_evaluate_models_writes_reports_and_models(tmp_path: Path) -> None:
    train = _make_model_frame(rows_per_class=10)
    test = _make_model_frame(rows_per_class=4)
    config = ModelingConfig(
        output_root=tmp_path,
        rf_n_estimators=20,
        rf_n_jobs=1,
        svm_sample_size=18,
        mlp_max_iter=30,
        mlp_n_iter_no_change=5,
        plot_dpi=80,
        label_order=("正常", "汽蚀", "轴承故障"),
    )

    summary = train_and_evaluate_models(train, test, ["feature_a", "feature_b", "feature_c", "feature_d"], config)

    assert set(summary["model"]) == {"RandomForest", "SVM", "MLP"}
    for filename in [
        "RandomForest.joblib",
        "SVM.joblib",
        "MLP.joblib",
        "model_metrics.csv",
        "RandomForest_confusion_matrix.csv",
        "SVM_confusion_matrix.csv",
        "MLP_confusion_matrix.csv",
    ]:
        assert (tmp_path / filename).exists()
