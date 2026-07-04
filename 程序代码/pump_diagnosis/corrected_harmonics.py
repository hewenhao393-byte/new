from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import BASE_RESULTS_ROOT, ModelingConfig, PipelineConfig
from pump_diagnosis.features import FEATURE_COLUMNS, extract_candidate_features
from pump_diagnosis.metadata import ConditionCatalog, build_file_index
from pump_diagnosis.pipeline import METADATA_COLUMNS
from pump_diagnosis.signal_processing import iter_windows, preprocess_run


HARMONIC_ALIAS_FEATURES = [
    "rot_1x_amp",
    "rot_2x_amp",
    "rot_3x_amp",
    "amp_2x_div_1x",
    "amp_3x_div_1x",
    "harmonic_energy_1x_3x",
]

CORRECTED_FEATURE_COLUMNS = FEATURE_COLUMNS + HARMONIC_ALIAS_FEATURES

CORRECTED_PROTECTED_FEATURES = {
    "rms",
    "kurtosis",
    "crest_factor",
    "amp_1x",
    "amp_2x",
    "amp_3x",
    "rot_1x_amp",
    "rot_2x_amp",
    "rot_3x_amp",
    "energy_1x",
    "energy_2x",
    "energy_3x",
    "amp_2x_div_1x",
    "amp_3x_div_1x",
    "harmonic_energy_1x_3x",
    "rotation_band_energy",
    "env_rms",
    "env_kurtosis",
    "wp_total_entropy",
    "band_ratio_10_500",
    "band_ratio_500_1000",
    "band_ratio_1000_2000",
    "band_ratio_2000_5000",
    "env_ratio_0_100",
    "env_ratio_100_500",
    "env_ratio_500_1000",
    "env_ratio_1000_3000",
}


def build_corrected_manifests(
    feature_root: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_manifest = pd.read_csv(feature_root / "train_manifest.csv")
    test_manifest = pd.read_csv(feature_root / "test_manifest.csv")
    catalog = ConditionCatalog.from_workbook(config.condition_workbook)
    file_index = build_file_index(config, catalog=catalog)
    corrected_index = file_index.loc[:, ["file_path", "run_id", "rpm", "machine_id", "condition_id", "channel", "original_fs", "sha256"]]
    train_corrected = _repair_manifest(train_manifest, corrected_index)
    test_corrected = _repair_manifest(test_manifest, corrected_index)

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    train_corrected.to_csv(output_root / "train_manifest_corrected.csv", index=False)
    test_corrected.to_csv(output_root / "test_manifest_corrected.csv", index=False)

    report = {
        "source_file_overlap_count": int(len(set(train_corrected["file_path"]).intersection(test_corrected["file_path"]))),
        "run_id_overlap_count": int(len(set(train_corrected["run_id"]).intersection(test_corrected["run_id"]))),
        "sha256_overlap_count": int(len(set(train_corrected["sha256"]).intersection(test_corrected["sha256"]))),
        "train_rows": int(len(train_corrected)),
        "test_rows": int(len(test_corrected)),
        "train_files": int(train_corrected["file_path"].nunique()),
        "test_files": int(test_corrected["file_path"].nunique()),
    }
    (output_root / "split_report_corrected.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_corrected, test_corrected, report


def extract_corrected_features(
    manifest: pd.DataFrame,
    split_name: str,
    config: PipelineConfig,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for file_index, (file_path, file_rows) in enumerate(manifest.groupby("file_path", sort=True), start=1):
        raw = pd.read_csv(file_path)
        for row in file_rows.itertuples(index=False):
            samples = raw[str(row.source_column)].to_numpy(dtype=np.float64)
            if not np.isfinite(samples).all():
                quality_rows.append({"run_id": row.run_id, "reason": "raw_nonfinite"})
                continue
            processed = preprocess_run(samples, config)
            for window_index, (start, end, window) in enumerate(
                iter_windows(processed, config.window_size, config.step_size)
            ):
                feature_row = _build_corrected_feature_row(window, row.rpm, config)
                feature_row.update(
                    {
                        "sample_id": f"{row.run_id}_window_{window_index}",
                        "label": row.label,
                        "run_id": row.run_id,
                        "file_path": row.file_path,
                        "machine_id": row.machine_id,
                        "condition_id": row.condition_id,
                        "rpm": row.rpm,
                        "channel": row.channel,
                        "window_index": window_index,
                        "window_start": start / config.processed_fs,
                        "window_end": end / config.processed_fs,
                        "original_fs": row.original_fs,
                        "processed_fs": config.processed_fs,
                    }
                )
                rows.append(feature_row)
        if file_index % 50 == 0:
            print(f"{split_name} corrected feature extraction: {file_index}/{manifest['file_path'].nunique()}")

    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS + CORRECTED_FEATURE_COLUMNS)
    frame.to_csv(output_path, index=False)
    quality_path = output_path.with_name(output_path.name.replace("features_raw_corrected.csv", "quality_report_corrected.csv"))
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
    return frame


def run_corrected_feature_extraction(
    feature_root: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_manifest, test_manifest, _ = build_corrected_manifests(feature_root, config)
    output_root = config.output_root
    train = extract_corrected_features(train_manifest, "train", config, output_root / "train_features_raw_corrected.csv")
    test = extract_corrected_features(test_manifest, "test", config, output_root / "test_features_raw_corrected.csv")
    return train, test


def build_feature_quality_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    combined = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        series = combined[column]
        finite_mask = np.isfinite(series.to_numpy(dtype=np.float64))
        rows.append(
            {
                "feature": column,
                "train_nunique": int(train[column].nunique(dropna=False)),
                "test_nunique": int(test[column].nunique(dropna=False)),
                "total_nunique": int(series.nunique(dropna=False)),
                "train_nan_count": int(train[column].isna().sum()),
                "test_nan_count": int(test[column].isna().sum()),
                "train_inf_count": int(np.isinf(train[column].to_numpy(dtype=np.float64)).sum()),
                "test_inf_count": int(np.isinf(test[column].to_numpy(dtype=np.float64)).sum()),
                "train_constant": bool(train[column].nunique(dropna=False) <= 1),
                "test_constant": bool(test[column].nunique(dropna=False) <= 1),
                "all_finite": bool(finite_mask.all()),
                "mean": float(np.nanmean(series.to_numpy(dtype=np.float64))),
                "std": float(np.nanstd(series.to_numpy(dtype=np.float64))),
            }
        )
    return pd.DataFrame(rows)


def remove_invalid_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    constant_threshold: float = 0.0,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    removed_rows: list[dict[str, object]] = []
    kept = list(feature_columns)
    for feature in list(feature_columns):
        train_values = train[feature].to_numpy(dtype=np.float64)
        test_values = test[feature].to_numpy(dtype=np.float64)
        train_finite = np.isfinite(train_values)
        test_finite = np.isfinite(test_values)
        if not train_finite.any():
            kept.remove(feature)
            removed_rows.append({"feature": feature, "reason": "train_all_nan_or_inf"})
            continue
        if np.nanstd(train_values[train_finite]) <= constant_threshold:
            kept.remove(feature)
            removed_rows.append({"feature": feature, "reason": "train_constant"})
            continue
        if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
            reason = "contains_nan_or_inf"
            if feature not in removed_rows:
                removed_rows.append({"feature": feature, "reason": reason})
    train_valid = train.loc[:, kept].copy()
    test_valid = test.loc[:, kept].copy()
    removed = pd.DataFrame(removed_rows).drop_duplicates(subset=["feature"], keep="first")
    return kept, train_valid, test_valid, removed


def select_correlated_features(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    threshold: float = 0.95,
    protected_features: set[str] | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protected = set(protected_features or CORRECTED_PROTECTED_FEATURES)
    kept = list(feature_columns)
    pair_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []

    while True:
        corr = train[kept].corr(method="pearson").abs()
        pairs = [
            (left, right, float(corr.loc[left, right]))
            for i, left in enumerate(kept)
            for right in kept[i + 1 :]
            if float(corr.loc[left, right]) >= threshold
        ]
        if not pairs:
            break
        pairs.sort(key=lambda item: item[2], reverse=True)
        left, right, value = pairs[0]
        left_score = _correlation_keep_score(train[left], left, protected, corr)
        right_score = _correlation_keep_score(train[right], right, protected, corr)
        if left in protected and right in protected:
            pair_rows.append({"left": left, "right": right, "correlation": value, "action": "retained_protected"})
            break
        if left_score <= right_score:
            keep_feature, remove_feature = left, right
        else:
            keep_feature, remove_feature = right, left
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "correlation": value,
                "action": "removed",
                "kept_feature": keep_feature,
                "removed_feature": remove_feature,
                "keep_score": json.dumps(left_score if left_score <= right_score else right_score, ensure_ascii=False),
                "remove_score": json.dumps(right_score if left_score <= right_score else left_score, ensure_ascii=False),
            }
        )
        kept.remove(remove_feature)
        removed_rows.append({"feature": remove_feature, "reason": f"correlated_abs_r_ge_{threshold}"})

    before = train[feature_columns].corr(method="pearson")
    after = train[kept].corr(method="pearson")
    pair_frame = pd.DataFrame(pair_rows)
    removed_frame = pd.DataFrame(removed_rows).drop_duplicates(subset=["feature"], keep="first")
    return kept, pair_frame, removed_frame, pd.concat([before.stack().rename("correlation_before").reset_index(), after.stack().rename("correlation_after").reset_index()], axis=1)


def fit_scaler_and_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    output_root: Path,
) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    scaler.fit(train[selected_features])
    train_scaled.loc[:, selected_features] = scaler.transform(train[selected_features])
    test_scaled.loc[:, selected_features] = scaler.transform(test[selected_features])
    joblib.dump(scaler, output_root / "scaler_corrected.joblib")
    train_scaled.to_csv(output_root / "train_features_scaled_corrected.csv", index=False)
    test_scaled.to_csv(output_root / "test_features_scaled_corrected.csv", index=False)
    return scaler, train_scaled, test_scaled


@dataclass(frozen=True)
class TunedModelResult:
    model_name: str
    best_params: dict[str, object]
    cv_results: pd.DataFrame
    test_metrics: dict[str, object]
    model: object


def tune_and_evaluate_model(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    config: ModelingConfig,
    *,
    groups: pd.Series,
    output_root: Path,
    scale: bool,
) -> TunedModelResult:
    output_root.mkdir(parents=True, exist_ok=True)
    param_grid = _parameter_grid_for_model(model_name, config)
    splitter = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    tune_train = _sample_training_frame(train, config.svm_sample_size if model_name in {"SVM", "MLP"} else None, config.random_state)
    x_train = tune_train[feature_columns]
    y_train = tune_train["label"]
    x_test = test[feature_columns]
    y_test = test["label"]

    cv_rows: list[dict[str, object]] = []
    best_score = -np.inf
    best_params: dict[str, object] | None = None

    for params in param_grid:
        print(f"{model_name} tuning params: {json.dumps(params, ensure_ascii=False)}")
        fold_scores: list[float] = []
        tune_groups = tune_train["file_path"] if "file_path" in tune_train.columns else groups.loc[tune_train.index]
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y_train, tune_groups)):
            fold_train_x = x_train.iloc[train_idx]
            fold_train_y = y_train.iloc[train_idx]
            fold_valid_x = x_train.iloc[valid_idx]
            fold_valid_y = y_train.iloc[valid_idx]
            model = _build_model(model_name, params, config, scale=scale)
            model.fit(fold_train_x, fold_train_y)
            predictions = model.predict(fold_valid_x)
            score = float(f1_score(fold_valid_y, predictions, average="macro"))
            fold_scores.append(score)
            cv_rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False),
                    "fold": fold_index,
                    "macro_f1": score,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                }
            )
        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    best_model = _build_model(model_name, best_params, config, scale=scale)
    best_model.fit(x_train, y_train)
    test_pred = best_model.predict(x_test)
    test_metrics = _summarize_predictions(y_test, test_pred, config.label_order)
    test_metrics.update(
        {
            "model": model_name,
            "best_params": best_params,
            "cv_macro_f1": best_score,
            "feature_count": len(feature_columns),
        }
    )

    pd.DataFrame(cv_rows).to_csv(output_root / f"{model_name.lower()}_cv_results_corrected.csv", index=False)
    _write_model_artifacts(output_root / model_name.lower(), model_name, best_model, y_test, test_pred, config.label_order, config.plot_dpi)
    (output_root / f"{model_name.lower()}_best_params_corrected.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([test_metrics]).to_csv(output_root / f"{model_name.lower()}_test_metrics_corrected.csv", index=False)
    return TunedModelResult(model_name=model_name, best_params=best_params, cv_results=pd.DataFrame(cv_rows), test_metrics=test_metrics, model=best_model)


def run_harmonic_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_feature_columns: list[str],
    config: ModelingConfig,
    *,
    output_root: Path,
) -> pd.DataFrame:
    groups = train["file_path"]
    harmonic_groups = {
        "A_no_harmonics": [
            feature
            for feature in base_feature_columns
            if feature not in {"rot_1x_amp", "rot_2x_amp", "rot_3x_amp", "amp_2x_div_1x", "amp_3x_div_1x", "harmonic_energy_1x_3x"}
        ],
        "B_only_1x": [
            feature
            for feature in base_feature_columns
            if feature not in {"rot_2x_amp", "rot_3x_amp", "amp_3x_div_1x", "harmonic_energy_1x_3x"}
        ],
        "C_full_harmonics": list(base_feature_columns),
    }
    rows: list[dict[str, object]] = []
    for name, features in harmonic_groups.items():
        result = tune_and_evaluate_model(
            "SVM",
            train,
            test,
            features,
            config,
            groups=groups,
            output_root=output_root / name,
            scale=True,
        )
        row = {
            "ablation": name,
            "feature_count": len(features),
            "macro_f1": result.test_metrics["macro_f1"],
            "balanced_accuracy": result.test_metrics["balanced_accuracy"],
            "unbalance_recall": result.test_metrics.get("recall_转子不平衡", 0.0),
            "misalignment_recall": result.test_metrics.get("recall_联轴器不对中", 0.0),
            "looseness_recall": result.test_metrics.get("recall_松动", 0.0),
        }
        rows.append(row)
    ablation = pd.DataFrame(rows)
    ablation.to_csv(output_root / "harmonic_feature_ablation.csv", index=False)
    return ablation


def summarize_compare_old_new(
    new_results: pd.DataFrame,
    old_results: pd.DataFrame,
    *,
    output_root: Path,
) -> pd.DataFrame:
    merged = new_results.merge(old_results, on="model", how="left", suffixes=("_new", "_old"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "model": row["model"],
                "version": "new",
                "feature_count": row.get("feature_count_new", np.nan),
                "accuracy": row.get("accuracy_new", np.nan),
                "balanced_accuracy": row.get("balanced_accuracy_new", np.nan),
                "macro_f1": row.get("macro_f1_new", np.nan),
                "normal_recall": row.get("recall_正常_new", np.nan),
                "unbalance_recall": row.get("recall_转子不平衡_new", np.nan),
                "misalignment_recall": row.get("recall_联轴器不对中_new", np.nan),
                "looseness_precision": row.get("precision_松动_new", np.nan),
                "looseness_recall": row.get("recall_松动_new", np.nan),
                "bearing_recall": row.get("recall_轴承故障_new", np.nan),
                "cavitation_recall": row.get("recall_汽蚀_new", np.nan),
            }
        )
        rows.append(
            {
                "model": row["model"],
                "version": "old",
                "feature_count": row.get("feature_count_old", np.nan),
                "accuracy": row.get("accuracy_old", np.nan),
                "balanced_accuracy": row.get("balanced_accuracy_old", np.nan),
                "macro_f1": row.get("macro_f1_old", np.nan),
                "normal_recall": row.get("recall_正常_old", np.nan),
                "unbalance_recall": row.get("recall_转子不平衡_old", np.nan),
                "misalignment_recall": row.get("recall_联轴器不对中_old", np.nan),
                "looseness_precision": row.get("precision_松动_old", np.nan),
                "looseness_recall": row.get("recall_松动_old", np.nan),
                "bearing_recall": row.get("recall_轴承故障_old", np.nan),
                "cavitation_recall": row.get("recall_汽蚀_old", np.nan),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "harmonic_fix_model_comparison.csv", index=False)
    return comparison


def _parameter_grid_for_model(model_name: str, config: ModelingConfig) -> list[dict[str, object]]:
    if model_name == "SVM":
        return [
            {"C": c, "gamma": "scale", "kernel": "rbf", "class_weight": "balanced"}
            for c in [1.0, 10.0, 100.0]
        ]
    if model_name == "MLP":
        return [
            {"hidden_layer_sizes": h, "alpha": alpha}
            for h, alpha in itertools.product([(64, 32), (128, 64)], [0.0001, 0.001])
        ]
    if model_name == "RandomForest":
        return [
            {"n_estimators": n, "max_depth": d, "max_features": mf}
            for n, d, mf in itertools.product([300], [None, 20], ["sqrt", "log2"])
        ]
    raise ValueError(f"unknown model: {model_name}")


def _build_model(model_name: str, params: dict[str, object], config: ModelingConfig, *, scale: bool) -> object:
    if model_name == "SVM":
        estimator = SVC(random_state=config.random_state, **params)
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)]) if scale else estimator
    if model_name == "MLP":
        estimator = MLPClassifier(
            activation=config.mlp_activation,
            solver=config.mlp_solver,
            max_iter=config.mlp_max_iter,
            early_stopping=False,
            random_state=config.random_state,
            **params,
        )
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)]) if scale else estimator
    if model_name == "RandomForest":
        return RandomForestClassifier(
            random_state=config.random_state,
            class_weight=config.rf_class_weight,
            n_jobs=config.rf_n_jobs,
            **params,
        )
    raise ValueError(f"unknown model: {model_name}")


def _summarize_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(labels), zero_division=0, output_dict=True)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        **{f"precision_{label}": float(p) for label, p in zip(labels, precision)},
        **{f"recall_{label}": float(r) for label, r in zip(labels, recall)},
        **{f"f1_{label}": float(v) for label, v in zip(labels, f1)},
        **{f"support_{label}": int(v) for label, v in zip(labels, support)},
    }


def _sample_training_frame(frame: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or len(frame) <= sample_size:
        return frame.copy().reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for label, group in frame.groupby("label"):
        target = max(1, int(round(sample_size * len(group) / len(frame))))
        parts.append(group.sample(n=min(target, len(group)), random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)
    return sampled.reset_index(drop=True)


def _write_model_artifacts(
    output_dir: Path,
    model_name: str,
    model: object,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{model_name}.joblib")
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=dpi)
    plt.close(fig)


def run_corrected_harmonics_pipeline(
    feature_root: Path,
    config: PipelineConfig,
    modeling_config: ModelingConfig,
) -> dict[str, object]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    train_manifest, test_manifest, split_report = build_corrected_manifests(feature_root, config)
    train_raw, test_raw = run_corrected_feature_extraction(feature_root, config)

    quality_report = build_feature_quality_report(train_raw, test_raw, CORRECTED_FEATURE_COLUMNS)
    quality_report.to_csv(output_root / "feature_quality_report_corrected.csv", index=False)
    pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(train_raw), len(test_raw)],
            "files": [train_raw["file_path"].nunique(), test_raw["file_path"].nunique()],
            "labels": [json.dumps(train_raw["label"].value_counts().to_dict(), ensure_ascii=False), json.dumps(test_raw["label"].value_counts().to_dict(), ensure_ascii=False)],
        }
    ).to_csv(output_root / "split_quality_corrected.csv", index=False)
    pd.DataFrame(
        {
            "check": ["source_file_overlap", "run_id_overlap"],
            "passed": [
                set(train_raw["file_path"]).isdisjoint(test_raw["file_path"]),
                set(train_raw["run_id"]).isdisjoint(test_raw["run_id"]),
            ],
            "overlap_count": [
                len(set(train_raw["file_path"]).intersection(test_raw["file_path"])),
                len(set(train_raw["run_id"]).intersection(test_raw["run_id"])),
            ],
        }
    ).to_csv(output_root / "data_leakage_check_corrected.csv", index=False)

    kept_valid_features, train_valid, test_valid, removed_invalid = remove_invalid_features(
        train_raw[CORRECTED_FEATURE_COLUMNS],
        test_raw[CORRECTED_FEATURE_COLUMNS],
        CORRECTED_FEATURE_COLUMNS,
    )
    removed_invalid.to_csv(output_root / "removed_invalid_features_corrected.csv", index=False)
    train_valid_full = pd.concat([train_raw[METADATA_COLUMNS], train_valid], axis=1)
    test_valid_full = pd.concat([test_raw[METADATA_COLUMNS], test_valid], axis=1)
    train_valid_full.to_csv(output_root / "train_features_valid_corrected.csv", index=False)
    test_valid_full.to_csv(output_root / "test_features_valid_corrected.csv", index=False)

    selected_features, high_corr_pairs, removed_corr, corr_matrix = select_correlated_features(
        train_valid,
        kept_valid_features,
        threshold=0.95,
    )
    high_corr_pairs.to_csv(output_root / "high_correlation_pairs_corrected.csv", index=False)
    removed_corr.to_csv(output_root / "removed_correlated_features_corrected.csv", index=False)
    (output_root / "selected_feature_columns_corrected.json").write_text(
        json.dumps(selected_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_selected = pd.concat([train_raw[METADATA_COLUMNS], train_raw[selected_features]], axis=1)
    test_selected = pd.concat([test_raw[METADATA_COLUMNS], test_raw[selected_features]], axis=1)
    train_selected.to_csv(output_root / "train_features_selected_corrected.csv", index=False)
    test_selected.to_csv(output_root / "test_features_selected_corrected.csv", index=False)

    _save_heatmap(train_valid, output_root / "correlation_heatmap_before_corrected.png")
    _save_heatmap(train_raw[selected_features], output_root / "correlation_heatmap_after_corrected.png")

    scaler, train_scaled, test_scaled = fit_scaler_and_transform(train_selected, test_selected, selected_features, output_root)

    modeling_root = output_root / "modeling_corrected"
    modeling_root.mkdir(parents=True, exist_ok=True)
    svm_result = tune_and_evaluate_model(
        "SVM",
        train_selected,
        test_selected,
        selected_features,
        modeling_config,
        groups=train_selected["file_path"],
        output_root=modeling_root / "svm",
        scale=True,
    )
    mlp_result = tune_and_evaluate_model(
        "MLP",
        train_selected,
        test_selected,
        selected_features,
        modeling_config,
        groups=train_selected["file_path"],
        output_root=modeling_root / "mlp",
        scale=True,
    )
    rf_result = tune_and_evaluate_model(
        "RandomForest",
        train_selected,
        test_selected,
        selected_features,
        modeling_config,
        groups=train_selected["file_path"],
        output_root=modeling_root / "random_forest",
        scale=False,
    )
    model_results = pd.DataFrame([svm_result.test_metrics, mlp_result.test_metrics, rf_result.test_metrics])
    model_results.to_csv(modeling_root / "model_metrics_corrected.csv", index=False)

    ablation = run_harmonic_ablation(train_selected, test_selected, selected_features, modeling_config, output_root=modeling_root / "ablation")

    old_metrics_path = BASE_RESULTS_ROOT / "modeling" / "model_metrics.csv"
    old_metrics = pd.read_csv(old_metrics_path) if old_metrics_path.exists() else pd.DataFrame()
    comparison = summarize_compare_old_new(model_results, _normalize_old_metrics(old_metrics), output_root=output_root)

    summary = {
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "selected_features": selected_features,
        "svm_best_params": svm_result.best_params,
        "mlp_best_params": mlp_result.best_params,
        "random_forest_best_params": rf_result.best_params,
        "split_report": split_report,
        "model_rows": len(model_results),
        "ablation_rows": len(ablation),
        "comparison_rows": len(comparison),
    }
    (output_root / "run_complete_corrected.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _normalize_old_metrics(old_metrics: pd.DataFrame) -> pd.DataFrame:
    if old_metrics.empty:
        return pd.DataFrame()
    frame = old_metrics.copy()
    for label in ("正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"):
        precision_col = f"precision_{label}"
        recall_col = f"recall_{label}"
        if precision_col not in frame.columns:
            frame[precision_col] = np.nan
        if recall_col not in frame.columns:
            frame[recall_col] = np.nan
    if "model" not in frame.columns:
        frame["model"] = frame.index.astype(str)
    if "feature_count" not in frame.columns:
        frame["feature_count"] = np.nan
    return frame


def _save_heatmap(frame: pd.DataFrame, path: Path) -> None:
    corr = frame.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _repair_manifest(manifest: pd.DataFrame, corrected_index: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["file_path", "run_id"]
    keep_columns = [column for column in corrected_index.columns if column not in {"machine_id", "condition_id", "channel", "original_fs", "sha256"}]
    corrected = manifest.drop(columns=[column for column in ("rpm", "machine_id", "condition_id", "channel", "original_fs", "sha256") if column in manifest.columns], errors="ignore")
    corrected = corrected.merge(corrected_index, on=join_keys, how="left", suffixes=("", "_fixed"))
    if "machine_id_fixed" in corrected.columns:
        corrected["machine_id"] = corrected["machine_id_fixed"]
        corrected = corrected.drop(columns=["machine_id_fixed"])
    if "condition_id_fixed" in corrected.columns:
        corrected["condition_id"] = corrected["condition_id_fixed"]
        corrected = corrected.drop(columns=["condition_id_fixed"])
    if "channel_fixed" in corrected.columns:
        corrected["channel"] = corrected["channel_fixed"]
        corrected = corrected.drop(columns=["channel_fixed"])
    if "original_fs_fixed" in corrected.columns:
        corrected["original_fs"] = corrected["original_fs_fixed"]
        corrected = corrected.drop(columns=["original_fs_fixed"])
    if "sha256_fixed" in corrected.columns:
        corrected["sha256"] = corrected["sha256_fixed"]
        corrected = corrected.drop(columns=["sha256_fixed"])
    if "rpm_fixed" in corrected.columns:
        corrected["rpm"] = corrected["rpm_fixed"]
        corrected = corrected.drop(columns=["rpm_fixed"])
    if "rpm" in corrected.columns:
        corrected["rpm"] = corrected["rpm"].astype(float)
    return corrected


def _build_corrected_feature_row(window: np.ndarray, rpm: float | None, config: PipelineConfig) -> OrderedDict[str, float]:
    base = extract_candidate_features(window, rpm=rpm, config=config)
    features = OrderedDict(base)
    features["rot_1x_amp"] = float(features["amp_1x"])
    features["rot_2x_amp"] = float(features["amp_2x"])
    features["rot_3x_amp"] = float(features["amp_3x"])
    features["amp_2x_div_1x"] = float(features["ratio_2x_to_1x"])
    features["amp_3x_div_1x"] = float(features["ratio_3x_to_1x"])
    features["harmonic_energy_1x_3x"] = float(features["rotation_band_energy"])
    return features


def _correlation_keep_score(
    series: pd.Series,
    feature_name: str,
    protected: set[str],
    corr: pd.DataFrame,
) -> tuple[float, float, float, float]:
    is_protected = 0.0 if feature_name in protected else 1.0
    missing_rate = float(series.isna().mean())
    variance = float(series.var())
    avg_corr = float(corr[feature_name].drop(index=feature_name, errors="ignore").mean()) if len(corr.columns) > 1 else 0.0
    physical_rank = 0.0 if feature_name in CORRECTED_PROTECTED_FEATURES else 1.0
    return (is_protected + physical_rank, missing_rate, -variance, avg_corr)
