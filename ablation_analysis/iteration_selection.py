"""Training-only CatBoost iteration selection at record level."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedGroupKFold

from baseline_analysis.evaluation import evaluate_predictions, fuse_records

from .config import CV_SPLITS, ITERATION_GRID, LABEL_ORDER, MAX_ITERATIONS, MODEL_PARAMS


_FUSION_META = ["record_id", "label", "motor", "rpm", "condition", "state", "severity"]
_RECORD_CONTRACT = ["label", "motor", "rpm", "condition", "state", "severity"]


def _one_dimensional(values, name: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    if series.ndim != 1 or series.empty:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if series.isna().any():
        raise ValueError(f"{name} must not contain null values")
    return series


def _manifest_hash(manifest: pd.DataFrame) -> str:
    records = manifest[["fold", "role", "record_id"]].to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_record_folds(labels, record_ids, n_splits: int = 5, seed: int = 2026):
    """Create deterministic grouped folds without accepting any feature or test data."""
    labels = _one_dimensional(labels, "labels")
    record_ids = _one_dimensional(record_ids, "record_ids")
    if len(labels) != len(record_ids):
        raise ValueError("labels and record_ids must have aligned lengths")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")

    pairs = pd.DataFrame({"label": labels, "record_id": record_ids})
    labels_per_record = pairs.groupby("record_id", sort=False, dropna=False)["label"].nunique(dropna=False)
    if not labels_per_record.eq(1).all():
        raise ValueError("each record_id must have exactly one label")
    record_labels = pairs.drop_duplicates("record_id")
    class_record_counts = record_labels.groupby("label", dropna=False)["record_id"].nunique()
    if (class_record_counts < n_splits).any():
        raise ValueError("not enough unique records per class for n_splits")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = []
    manifest_rows = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    for fold, (fit_idx, validation_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups=record_ids), start=1
    ):
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        validation_idx = np.asarray(validation_idx, dtype=np.int64)
        fit_records = set(record_ids.iloc[fit_idx].tolist())
        validation_records = set(record_ids.iloc[validation_idx].tolist())
        if fit_records & validation_records:
            raise RuntimeError("record overlap detected between fit and validation")
        validation_counts[validation_idx] += 1
        fold_indices.append((fit_idx, validation_idx))
        for role, ids in (("fit", fit_records), ("validation", validation_records)):
            for record_id in sorted(ids, key=lambda value: str(value)):
                manifest_rows.append({"fold": fold, "role": role, "record_id": record_id})
    if not np.all(validation_counts == 1):
        raise RuntimeError("every row must appear in validation exactly once")

    manifest = pd.DataFrame(manifest_rows, columns=["fold", "role", "record_id"])
    fold_hash = _manifest_hash(manifest)
    manifest["fold_sha256"] = fold_hash
    return manifest, fold_indices, fold_hash


def _validate_checkpoints(checkpoints: Sequence[int]) -> list[int]:
    values = list(checkpoints)
    if not values or any(not isinstance(value, (int, np.integer)) or value < 1 for value in values):
        raise ValueError("checkpoints must be positive one-based integers")
    values = [int(value) for value in values]
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("checkpoints must be unique and sorted")
    return values


def _validate_classes(model, classes: Sequence[str]) -> list[str]:
    classes = list(classes)
    if len(classes) != len(set(classes)) or set(classes) != set(LABEL_ORDER):
        raise ValueError("classes must contain the fixed LABEL_ORDER exactly once")
    model_classes = list(getattr(model, "classes_", []))
    if model_classes != classes:
        raise ValueError("model classes must match supplied classes in probability-column order")
    return classes


def _validate_record_contract(frame: pd.DataFrame, name: str) -> None:
    """Require one label/metadata tuple per record, treating null as a value."""
    conflicts = []
    for record_id, group in frame.groupby("record_id", sort=False, dropna=False):
        conflicting_columns = []
        for column in _RECORD_CONTRACT:
            first = group[column].iloc[0]
            matches = group[column].isna() if pd.isna(first) else group[column].eq(first)
            if not matches.all():
                conflicting_columns.append(column)
        if conflicting_columns:
            conflicts.append((record_id, conflicting_columns))
    if conflicts:
        record_id, columns = conflicts[0]
        raise ValueError(
            f"{name} record consistency conflict for record_id={record_id!r}; "
            f"conflicting columns: {', '.join(columns)}"
        )


def staged_record_scores(model, validation, feature_names, checkpoints, classes):
    """Score selected one-based stages from one CatBoost staged prediction stream."""
    feature_names = list(feature_names)
    checkpoints = _validate_checkpoints(checkpoints)
    classes = _validate_classes(model, classes)
    required = list(dict.fromkeys(_FUSION_META + feature_names))
    missing = [column for column in required if column not in validation.columns]
    if missing:
        raise ValueError(f"validation missing required columns: {missing}")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique")
    unknown = sorted(set(validation["label"].dropna()) - set(LABEL_ORDER))
    if unknown:
        raise ValueError(f"validation contains unknown label values: {unknown}")
    _validate_record_contract(validation, "validation")
    nonnull = ["record_id", "label", *feature_names]
    if validation[nonnull].isna().any().any():
        raise ValueError("validation record_id, label, and features must not contain null values")

    wanted = set(checkpoints)
    rows = []
    seen_last = 0
    stream: Iterable[np.ndarray] = model.staged_predict_proba(validation.loc[:, feature_names])
    for iteration, probabilities in enumerate(stream, start=1):
        seen_last = iteration
        if iteration not in wanted:
            if iteration >= checkpoints[-1]:
                break
            continue
        probabilities = np.asarray(probabilities, dtype=float)
        expected_shape = (len(validation), len(classes))
        if probabilities.shape != expected_shape:
            raise ValueError(f"probability shape must be {expected_shape}, got {probabilities.shape}")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0)
        ):
            raise ValueError("probabilities must be finite values in [0, 1] with rows summing to one")
        probability_frame = validation.loc[:, _FUSION_META].copy()
        for class_index, label in enumerate(classes):
            probability_frame[label] = probabilities[:, class_index]
        probability_frame["predicted_label"] = [classes[index] for index in probabilities.argmax(axis=1)]
        fused = fuse_records(probability_frame, classes)
        metrics = evaluate_predictions(fused["label"], fused["predicted_label"], LABEL_ORDER)
        rows.append({"iteration": iteration, "record_macro_f1": metrics["macro_f1"]})
        if iteration >= checkpoints[-1]:
            break
    if seen_last < checkpoints[-1] or len(rows) != len(checkpoints):
        raise ValueError(
            f"incomplete checkpoint stream: required through {checkpoints[-1]}, received through {seen_last}"
        )
    return pd.DataFrame(rows, columns=["iteration", "record_macro_f1"])


def aggregate_cv_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "iteration", "record_macro_f1"}
    missing = sorted(required - set(fold_scores.columns))
    if missing:
        raise ValueError(f"fold scores missing required columns: {missing}")
    if fold_scores.empty:
        raise ValueError("fold scores must not be empty")
    numeric = fold_scores["record_macro_f1"].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("record_macro_f1 must be finite")
    if fold_scores.duplicated(["fold", "iteration"]).any():
        raise ValueError("duplicate fold and iteration scores")
    summary = (
        fold_scores.groupby("iteration", sort=True)["record_macro_f1"]
        .agg(mean_record_macro_f1="mean", std_record_macro_f1="std", fold_count="count")
        .reset_index()
    )
    return summary.sort_values("iteration", kind="stable").reset_index(drop=True)


def choose_iteration(summary, checkpoints=ITERATION_GRID, expected_folds=CV_SPLITS) -> int:
    checkpoints = _validate_checkpoints(checkpoints)
    required = {"iteration", "mean_record_macro_f1", "std_record_macro_f1", "fold_count"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary missing required columns: {missing}")
    if summary.empty:
        raise ValueError("summary must not be empty")
    if summary["iteration"].duplicated().any():
        raise ValueError("duplicate iterations in summary")
    if set(summary["iteration"]) != set(checkpoints):
        raise ValueError("summary iterations must match the checkpoint grid")
    if not summary["fold_count"].eq(expected_folds).all():
        raise ValueError("each iteration must contain the expected fold count")
    values = summary[["mean_record_macro_f1", "std_record_macro_f1"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("summary scores must be finite")
    maximum = float(summary["mean_record_macro_f1"].max())
    tied = summary[np.isclose(summary["mean_record_macro_f1"], maximum, atol=1e-12, rtol=0)]
    return int(tied["iteration"].min())


def _normalize_folds(folds, supplied_indices=None):
    if supplied_indices is None:
        if not isinstance(folds, tuple) or len(folds) != 3:
            raise ValueError("folds must be a manifest plus indices or the result from make_record_folds")
        manifest, indices, fold_hash = folds
    else:
        manifest, indices = folds, supplied_indices
        if not isinstance(manifest, pd.DataFrame):
            raise ValueError("fold manifest must be a DataFrame")
        fold_hash = _manifest_hash(manifest)
        if "fold_sha256" in manifest:
            stored_hashes = manifest["fold_sha256"].drop_duplicates().tolist()
            if stored_hashes != [fold_hash]:
                raise ValueError("fold manifest does not match its stored fold hash")
    if not isinstance(manifest, pd.DataFrame) or _manifest_hash(manifest) != fold_hash:
        raise ValueError("fold manifest does not match fold hash")
    return manifest, indices, fold_hash


def select_iterations(train, feature_names, folds, fold_indices=None, checkpoints=ITERATION_GRID):
    """Fit one max-iteration model per supplied training fold and select a stage."""
    feature_names = list(feature_names)
    checkpoints = _validate_checkpoints(checkpoints)
    if checkpoints[-1] > MAX_ITERATIONS:
        raise ValueError("checkpoint grid exceeds MAX_ITERATIONS")
    manifest, fold_indices, fold_hash = _normalize_folds(folds, fold_indices)
    required = list(dict.fromkeys(_FUSION_META + feature_names))
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"train missing required columns: {missing}")
    unknown = sorted(set(train["label"].dropna()) - set(LABEL_ORDER))
    if unknown:
        raise ValueError(f"train contains unknown label values: {unknown}")
    _validate_record_contract(train, "train")
    nonnull = ["record_id", "label", *feature_names]
    if train[nonnull].isna().any().any():
        raise ValueError("train record_id, label, and features must not contain null values")

    all_rows = set(range(len(train)))
    validation_counts = np.zeros(len(train), dtype=np.int64)
    if len(fold_indices) != manifest["fold"].nunique():
        raise ValueError("fold indices do not match the fold manifest")
    for fold, (fit_idx, validation_idx) in enumerate(fold_indices, start=1):
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        validation_idx = np.asarray(validation_idx, dtype=np.int64)
        fit_rows = set(fit_idx.tolist())
        validation_rows = set(validation_idx.tolist())
        if (
            len(fit_rows) != len(fit_idx)
            or len(validation_rows) != len(validation_idx)
            or fit_rows & validation_rows
            or fit_rows | validation_rows != all_rows
        ):
            raise ValueError("fold indices do not form the partition declared by the fold manifest")
        expected_fit = set(manifest.loc[(manifest["fold"] == fold) & (manifest["role"] == "fit"), "record_id"])
        expected_validation = set(
            manifest.loc[(manifest["fold"] == fold) & (manifest["role"] == "validation"), "record_id"]
        )
        if (
            set(train.iloc[fit_idx]["record_id"]) != expected_fit
            or set(train.iloc[validation_idx]["record_id"]) != expected_validation
        ):
            raise ValueError("fold indices do not match record assignments in the fold manifest")
        validation_counts[validation_idx] += 1
    if not np.all(validation_counts == 1):
        raise ValueError("fold indices must place every training row in validation exactly once")

    rows = []
    params = {**MODEL_PARAMS, "iterations": MAX_ITERATIONS}
    for fold, (fit_idx, validation_idx) in enumerate(fold_indices, start=1):
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        validation_idx = np.asarray(validation_idx, dtype=np.int64)
        if set(train.iloc[fit_idx]["record_id"]) & set(train.iloc[validation_idx]["record_id"]):
            raise ValueError("record overlap in supplied fold indices")
        model = CatBoostClassifier(**params)
        model.fit(Pool(train.iloc[fit_idx][feature_names], train.iloc[fit_idx]["label"]))
        scores = staged_record_scores(
            model,
            train.iloc[validation_idx],
            feature_names,
            checkpoints,
            list(model.classes_),
        )
        scores.insert(0, "fold", fold)
        rows.append(scores)
    fold_scores = pd.concat(rows, ignore_index=True)
    summary = aggregate_cv_scores(fold_scores)
    selected_iteration = choose_iteration(summary, checkpoints, expected_folds=len(fold_indices))
    return {
        "fold_scores": fold_scores,
        "summary": summary,
        "selected_iteration": selected_iteration,
        "fold_hash": fold_hash,
    }
