from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .contracts import CHANNELS, CV_FOLDS, FEATURE_NAMES, LABELS, MODEL_PARAMS
from .data import AlignedData, WINDOW_KEY


class ModelingError(RuntimeError):
    """Raised when model outputs cannot satisfy the probability contract."""


def default_model_factory(params: dict):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params)


def _ordered_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame.loc[:, FEATURE_NAMES]), dtype=float)
    classes = tuple(str(value) for value in model.classes_)
    if set(classes) != set(LABELS):
        raise ModelingError(f"model classes drift: {classes!r}")
    ordered = raw[:, [classes.index(label) for label in LABELS]]
    if ordered.shape != (len(frame), len(LABELS)) or not np.allclose(ordered.sum(axis=1), 1.0):
        raise ModelingError("model returned invalid probabilities")
    return ordered


def _prediction_frame(frame: pd.DataFrame, channel: int, fold: int | None, probabilities: np.ndarray) -> pd.DataFrame:
    columns = list(WINDOW_KEY) + ["parent_group_id", "label"]
    result = frame.loc[:, columns].reset_index(drop=True).copy()
    result["channel"] = channel
    if fold is not None:
        result["fold"] = fold
    for index, label in enumerate(LABELS):
        result[f"prob_{label}"] = probabilities[:, index]
    return result


def generate_oof_predictions(
    data: AlignedData,
    *,
    model_factory: Callable[[dict], object] = default_model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, metadata = [], []
    for channel in CHANNELS:
        frame = data.train_dev[channel]
        for fold in range(1, CV_FOLDS + 1):
            validation_mask = frame["parent_group_id"].map(data.fold_by_parent).eq(fold)
            fit = frame.loc[~validation_mask]
            validation = frame.loc[validation_mask]
            if fit.empty or validation.empty:
                raise ModelingError(f"CH{channel} fold {fold} has an empty fit/validation partition")
            started = time.monotonic()
            params = dict(MODEL_PARAMS)
            model = model_factory(params)
            model.fit(fit.loc[:, FEATURE_NAMES], fit["label"])
            probabilities = _ordered_probabilities(model, validation)
            outputs.append(_prediction_frame(validation, channel, fold, probabilities))
            metadata.append({
                "channel": channel, "fold": fold, "fit_windows": len(fit),
                "validation_windows": len(validation), "fit_parents": fit["parent_group_id"].nunique(),
                "validation_parents": validation["parent_group_id"].nunique(),
                "elapsed_seconds": time.monotonic() - started,
            })
    predictions = pd.concat(outputs, ignore_index=True)
    expected = sum(len(data.train_dev[channel]) for channel in CHANNELS)
    if len(predictions) != expected or predictions.duplicated(["channel", *WINDOW_KEY]).any():
        raise ModelingError("OOF prediction coverage is not exactly once per channel/window")
    return predictions, pd.DataFrame(metadata)


def generate_strict_predictions(
    data: AlignedData,
    *,
    model_factory: Callable[[dict], object] = default_model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, metadata = [], []
    for channel in CHANNELS:
        fit = data.train_dev[channel]
        strict = data.strict[channel]
        started = time.monotonic()
        model = model_factory(dict(MODEL_PARAMS))
        model.fit(fit.loc[:, FEATURE_NAMES], fit["label"])
        outputs.append(_prediction_frame(strict, channel, None, _ordered_probabilities(model, strict)))
        metadata.append({
            "channel": channel, "fit_scope": "all_train_dev", "fit_windows": len(fit),
            "strict_windows": len(strict), "fit_parents": fit["parent_group_id"].nunique(),
            "strict_parents": strict["parent_group_id"].nunique(),
            "elapsed_seconds": time.monotonic() - started,
        })
    predictions = pd.concat(outputs, ignore_index=True)
    expected = sum(len(data.strict[channel]) for channel in CHANNELS)
    if len(predictions) != expected or predictions.duplicated(["channel", *WINDOW_KEY]).any():
        raise ModelingError("strict prediction coverage is not exactly once per channel/window")
    return predictions, pd.DataFrame(metadata)
