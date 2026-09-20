"""Final training and held-out evaluation for one ablation run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from baseline_analysis.config import MODEL_PARAMS
from baseline_analysis.evaluation import evaluate_predictions, fuse_records

from . import config as ablation_config
from .config import FEATURE_40, FEATURE_43, LABEL_ORDER


_META = ["record_id", "window_id", "label", "motor", "rpm", "condition", "state", "severity", "start_sample", "end_sample"]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _probability_frame(data: pd.DataFrame, probabilities: np.ndarray, classes: Sequence[str]) -> pd.DataFrame:
    output = data.loc[:, _META].copy()
    for index, label in enumerate(classes):
        output[label] = probabilities[:, index]
    output["predicted_label"] = [classes[index] for index in probabilities.argmax(axis=1)]
    output["max_class_probability"] = probabilities.max(axis=1)
    return output


def train_selected_model(
    data: pd.DataFrame,
    split_mode: str,
    channel: int,
    feature_names: Sequence[str],
    selected_iteration: int,
    fold_hash: str,
    fold_scores: pd.DataFrame,
    iteration_summary: pd.DataFrame,
    output_dir,
):
    """Fit once on the prescribed training split and evaluate once on test."""
    features = list(feature_names)
    if features not in (FEATURE_43, FEATURE_40):
        raise ValueError("feature_names must be exactly FEATURE_43 or FEATURE_40 in contract order")
    if selected_iteration not in ablation_config.ITERATION_GRID:
        raise ValueError("selected_iteration must belong to config.ITERATION_GRID")
    if split_mode not in {"record", "temporal"}:
        raise ValueError("split_mode must be record or temporal")

    training_split = "train_dev" if split_mode == "record" else "train"
    train = data.loc[data["split"].eq(training_split)].reset_index(drop=True)
    test = data.loc[data["split"].eq("test")].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("prescribed training and test splits must both be non-empty")
    if set(train["record_id"]) & set(test["record_id"]) and split_mode == "record":
        raise ValueError("record split contains train/test record overlap")
    if set(train["label"]) != set(LABEL_ORDER):
        raise ValueError("training split must contain every fixed class")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    fold_scores.to_csv(out / "internal_cv_fold_scores.csv", index=False, encoding="utf-8-sig")
    iteration_summary.to_csv(out / "internal_cv_iteration_summary.csv", index=False, encoding="utf-8-sig")

    parameters = {**MODEL_PARAMS, "iterations": int(selected_iteration)}
    model = CatBoostClassifier(**parameters)
    model.fit(Pool(train.loc[:, features], train["label"]))
    model_path = out / "model.cbm"
    model.save_model(str(model_path))

    probabilities = np.asarray(model.predict_proba(test.loc[:, features]), dtype=float)
    classes = list(model.classes_)
    if set(classes) != set(LABEL_ORDER):
        raise RuntimeError("trained model classes do not match the fixed label contract")
    predictions = _probability_frame(test, probabilities, classes)
    predictions.to_csv(out / "window_predictions.csv", index=False, encoding="utf-8-sig")
    records = fuse_records(predictions, classes)
    records.to_csv(out / "record_predictions.csv", index=False, encoding="utf-8-sig")

    window_metrics = evaluate_predictions(predictions["label"], predictions["predicted_label"], LABEL_ORDER)
    record_metrics = evaluate_predictions(records["label"], records["predicted_label"], LABEL_ORDER)
    _write_json(out / "window_metrics.json", window_metrics)
    _write_json(out / "record_metrics.json", record_metrics)

    importance = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
    importance = importance.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(out / "feature_importance.csv", index=False, encoding="utf-8-sig")

    restored = CatBoostClassifier()
    restored.load_model(str(model_path))
    restored_probabilities = np.asarray(restored.predict_proba(test.loc[:, features]), dtype=float)
    np.testing.assert_allclose(probabilities, restored_probabilities, atol=1e-12, rtol=0)

    metadata = {
        "split_mode": split_mode,
        "channel": int(channel),
        "training_split": training_split,
        "test_split": "test",
        "selected_iteration": int(selected_iteration),
        "parameters": parameters,
        "features": features,
        "classes": classes,
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "train_records": int(train["record_id"].nunique()),
        "test_records": int(test["record_id"].nunique()),
        "fold_sha256": fold_hash,
        "confidence": {
            "window_mean_max_class_probability": float(predictions["max_class_probability"].mean()),
            "record_mean_max_class_probability": float(records["mean_max_class_probability"].mean()),
            "window_prediction_count": int(len(predictions)),
            "record_prediction_count": int(len(records)),
        },
        "model_reload_probability_atol": 1e-12,
    }
    _write_json(out / "metadata.json", metadata)
    return {"window": window_metrics, "record": record_metrics, "metadata": metadata}
