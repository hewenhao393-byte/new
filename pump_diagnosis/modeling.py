from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from config import ModelingConfig


class LeakageError(RuntimeError):
    def __init__(self, report: dict[str, int]):
        super().__init__("training/test leakage detected")
        self.report = report


@dataclass(frozen=True)
class FeatureFilterResult:
    kept_features: list[str]
    removed_reasons: dict[str, str]


@dataclass(frozen=True)
class TopKResult:
    selected_k: int
    selected_features: list[str]
    summary: pd.DataFrame
    fold_results: pd.DataFrame


def audit_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    train_window_source = set(zip(train["run_id"], train["window_index"]))
    test_window_source = set(zip(test["run_id"], test["window_index"]))
    report = {
        "file_path_overlap_count": len(set(train["file_path"]).intersection(test["file_path"])),
        "run_id_overlap_count": len(set(train["run_id"]).intersection(test["run_id"])),
        "sample_id_overlap_count": len(set(train["sample_id"]).intersection(test["sample_id"])),
        "window_source_overlap_count": len(train_window_source.intersection(test_window_source)),
    }
    if any(report.values()):
        raise LeakageError(report)
    return report


def drop_nonfinite_feature_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_mask = np.isfinite(train[feature_columns].to_numpy()).all(axis=1)
    test_mask = np.isfinite(test[feature_columns].to_numpy()).all(axis=1)
    return (
        train.loc[train_mask].reset_index(drop=True),
        test.loc[test_mask].reset_index(drop=True),
        {
            "train_removed_rows": int((~train_mask).sum()),
            "test_removed_rows": int((~test_mask).sum()),
        },
    )


def fit_training_feature_filter(
    train_features: pd.DataFrame,
    feature_columns: list[str],
    config: ModelingConfig,
) -> FeatureFilterResult:
    kept = list(feature_columns)
    removed: dict[str, str] = {}

    for feature in list(kept):
        series = train_features[feature]
        if series.nunique(dropna=False) <= 1:
            kept.remove(feature)
            removed[feature] = "constant"
        elif float(series.var()) <= config.near_zero_variance_threshold:
            kept.remove(feature)
            removed[feature] = "near_zero_variance"

    corr = train_features[kept].corr().abs()
    while True:
        pairs = [
            (left, right, float(corr.loc[left, right]))
            for i, left in enumerate(kept)
            for right in kept[i + 1 :]
            if float(corr.loc[left, right]) >= config.correlation_threshold
        ]
        if not pairs:
            break
        left, right, _ = max(pairs, key=lambda item: item[2])
        to_remove = _pick_correlated_feature_to_remove(left, right, config)
        kept.remove(to_remove)
        removed[to_remove] = "correlated"
        corr = train_features[kept].corr().abs()

    return FeatureFilterResult(kept_features=kept, removed_reasons=removed)


def evaluate_top_k_grouped(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: ModelingConfig,
) -> TopKResult:
    splitter = StratifiedGroupKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    labels = frame["label"]
    groups = frame["file_path"]
    candidate_scores: list[dict[str, float]] = []
    fold_rows: list[dict[str, float]] = []
    ranked_features = list(feature_columns)

    for k in config.top_k_candidates:
        selected = ranked_features[:k]
        scores: list[float] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(frame[selected], labels, groups)):
            fold_train = frame.iloc[train_idx]
            fold_valid = frame.iloc[valid_idx]
            model = RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                random_state=config.random_state,
                n_jobs=config.rf_n_jobs,
            )
            model.fit(fold_train[selected], fold_train["label"])
            predictions = model.predict(fold_valid[selected])
            score = f1_score(fold_valid["label"], predictions, average="macro")
            scores.append(score)
            overlap = len(set(fold_train["file_path"]).intersection(fold_valid["file_path"]))
            fold_rows.append(
                {
                    "k": k,
                    "fold": fold_index,
                    "macro_f1": score,
                    "group_overlap_count": overlap,
                    "validation_file_count": fold_valid["file_path"].nunique(),
                }
            )
        candidate_scores.append({"k": k, "macro_f1_mean": float(np.mean(scores))})

    summary = pd.DataFrame(candidate_scores)
    selected_k = int(summary.sort_values(["macro_f1_mean", "k"], ascending=[False, True]).iloc[0]["k"])
    return TopKResult(
        selected_k=selected_k,
        selected_features=ranked_features[:selected_k],
        summary=summary,
        fold_results=pd.DataFrame(fold_rows),
    )


def _pick_correlated_feature_to_remove(left: str, right: str, config: ModelingConfig) -> str:
    for preferred in config.correlation_priority:
        if left == preferred:
            return right
        if right == preferred:
            return left
    return sorted([left, right])[1]
