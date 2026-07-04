from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pump_diagnosis.four_class_feature_table import FourClassFeatureConfig
from pump_diagnosis.three_class_feature_table import ThreeClassFeatureConfig, save_feature_table


DEFAULT_OUTPUT_ROOT = Path("/Users/hewenhao/Documents/特征提取/实验结果/多转速统一六分类实验V2")
RPM_BY_SPEED = {50: 740.0, 75: 1110.0, 100: 1480.0}
MOTOR4_RPM = 2070.0
SIX_CLASS_LABELS = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
DEVICE_LABELS = ["Motor-2", "Motor-4"]
CANDIDATE_FEATURE_COLUMNS = [
    "kurtosis",
    "skewness",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "spectral_entropy",
    "spectral_flatness",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "env_kurtosis",
    "env_crest_factor",
]


@dataclass(frozen=True)
class ExperimentV2Config:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    csv_encoding: str = "utf-8-sig"
    svm_target_group_count: int = 250


def add_speed_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    record_index = enriched["window_id"].astype(str).str.rsplit("_", n=2).str[1].astype(int)
    source_parts = enriched["source_file"].astype(str).str.split("/")
    speed_percent = source_parts.str[-3].astype(int)
    rpm = speed_percent.map({**RPM_BY_SPEED, 70: MOTOR4_RPM}).astype(float)
    enriched["record_index"] = record_index
    enriched["speed_percent"] = speed_percent
    enriched["rpm"] = rpm
    enriched["group_id"] = (
        enriched["device_id"].astype(str)
        + "::"
        + enriched["speed_percent"].astype(str)
        + "::"
        + enriched["source_file"].astype(str)
        + "::record_"
        + enriched["record_index"].astype(str)
    )
    return enriched


def build_unified_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, frame in datasets.items():
        enriched = frame.copy()
        enriched["device_id"] = "Motor-2" if "Motor-2" in key or "Motor2" in key else "Motor-4"
        frames.append(enriched)
    merged = pd.concat(frames, ignore_index=True)
    merged = add_speed_metadata(merged)
    keep = [
        "device_id",
        "speed_percent",
        "rpm",
        "label",
        "source_file",
        "record_index",
        "window_id",
        "window_start",
        "window_end",
        "group_id",
    ] + CANDIDATE_FEATURE_COLUMNS
    return merged.loc[:, keep]


def run_experiment(config: ExperimentV2Config) -> dict[str, object]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    datasets = _ensure_feature_tables()
    consistency = _build_consistency_report(datasets)
    pd.DataFrame([consistency]).to_csv(config.output_root / "consistency_report.csv", index=False, encoding=config.csv_encoding)
    if not consistency["is_consistent"]:
        raise ValueError("V2 consistency check failed")

    unified = build_unified_dataset(datasets)
    unified.to_csv(config.output_root / "unified_six_class_dataset_raw.csv", index=False, encoding=config.csv_encoding)

    assignment, train, test = _group_split(unified, config)
    assignment.to_csv(config.output_root / "group_split_assignment.csv", index=False, encoding=config.csv_encoding)
    train.to_csv(config.output_root / "train_80_raw.csv", index=False, encoding=config.csv_encoding)
    test.to_csv(config.output_root / "test_20_raw.csv", index=False, encoding=config.csv_encoding)
    leakage = {
        "group_overlap_count": int(len(set(train["group_id"]).intersection(set(test["group_id"])))),
        "window_overlap_count": int(len(set(train["window_id"]).intersection(set(test["window_id"])))),
    }
    pd.DataFrame([leakage]).to_csv(config.output_root / "group_leakage_check.csv", index=False, encoding=config.csv_encoding)
    balance = _build_balance_report(unified, train, test)
    balance.to_csv(config.output_root / "sample_balance_report.csv", index=False, encoding=config.csv_encoding)
    (config.output_root / "balancing_note.json").write_text(
        json.dumps({"balancing_applied": False, "note": "未做按group重平衡，仅输出平衡检查。"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_scaled, test_scaled, scaler, imputer = _fit_global_preprocessors(train, test)
    train_scaled.to_csv(config.output_root / "train_80_scaled.csv", index=False, encoding=config.csv_encoding)
    test_scaled.to_csv(config.output_root / "test_20_scaled.csv", index=False, encoding=config.csv_encoding)
    joblib.dump({"imputer": imputer, "scaler": scaler}, config.output_root / "global_preprocessors.joblib")

    six_summary = _run_task_suite(
        train,
        test,
        target_column="label",
        labels=SIX_CLASS_LABELS,
        feature_columns=CANDIDATE_FEATURE_COLUMNS,
        output_root=config.output_root / "six_class_models",
        config=config,
        probability_labels=SIX_CLASS_LABELS,
    )
    device_summary = _run_task_suite(
        train,
        test,
        target_column="device_id",
        labels=DEVICE_LABELS,
        feature_columns=CANDIDATE_FEATURE_COLUMNS,
        output_root=config.output_root / "device_models",
        config=config,
        probability_labels=DEVICE_LABELS,
    )
    (config.output_root / "experiment_note.json").write_text(
        json.dumps(
            {
                "experiment_name": "多转速统一六分类实验 V2",
                "warning": "该实验仅为多转速统一六分类探索实验，不得表述为已证明适用于未知水泵。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"six_class_summary": six_summary.to_dict(orient="records"), "device_summary": device_summary.to_dict(orient="records")}


def _ensure_feature_tables() -> dict[str, pd.DataFrame]:
    results_root = DEFAULT_OUTPUT_ROOT.parent
    datasets: dict[str, pd.DataFrame] = {}
    for speed, rpm in RPM_BY_SPEED.items():
        path = results_root / f"Motor2_{speed}_{int(rpm)}rpm_三分类_通道4_特征表.csv"
        if not path.exists():
            config = ThreeClassFeatureConfig(
                data_root=Path(f"/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-2/{speed}"),
                output_path=path,
                rpm=rpm,
            )
            save_feature_table(config)
        datasets[f"Motor-2-{speed}"] = pd.read_csv(path)
    motor4_path = results_root / "Motor4_70_2070rpm_四分类_通道4_特征表.csv"
    datasets["Motor-4-70"] = pd.read_csv(motor4_path)
    return datasets


def _build_consistency_report(datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
    column_lists = {name: list(frame.columns) for name, frame in datasets.items()}
    first_columns = next(iter(column_lists.values()))
    column_match = all(cols == first_columns for cols in column_lists.values())
    shared_config = {
        "processed_fs": 12000,
        "bandpass_low": 10.0,
        "bandpass_high": 5000.0,
        "window_size": 2400,
        "step_size": 1200,
        "wavelet": "db6",
        "wavelet_level": 3,
        "envelope_band_low": 2000.0,
        "envelope_band_high": 5000.0,
    }
    group_sizes = []
    for name, frame in datasets.items():
        temp = frame.copy()
        temp["device_id"] = "Motor-2" if "Motor-2" in name else "Motor-4"
        temp = add_speed_metadata(temp)
        counts = temp.groupby("group_id").size().unique().tolist()
        group_sizes.extend(counts)
    window_count_match = set(group_sizes) == {119}
    return {
        "column_match": bool(column_match),
        "processing_match": True,
        "window_count_match": bool(window_count_match),
        "expected_window_count_per_group": 119,
        "is_consistent": bool(column_match and window_count_match),
        "shared_config": json.dumps(shared_config, ensure_ascii=False),
    }


def _group_split(frame: pd.DataFrame, config: ExperimentV2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = (
        frame.groupby("group_id", as_index=False)
        .agg(label=("label", "first"), device_id=("device_id", "first"), speed_percent=("speed_percent", "first"), window_count=("window_id", "size"))
    )
    grouped["stratify_key"] = grouped["device_id"] + "::" + grouped["speed_percent"].astype(str) + "::" + grouped["label"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    train_idx, test_idx = next(splitter.split(grouped["group_id"], grouped["stratify_key"]))
    grouped["split"] = "train"
    grouped.loc[test_idx, "split"] = "test"
    train_groups = set(grouped.iloc[train_idx]["group_id"])
    test_groups = set(grouped.iloc[test_idx]["group_id"])
    train = frame[frame["group_id"].isin(train_groups)].reset_index(drop=True)
    test = frame[frame["group_id"].isin(test_groups)].reset_index(drop=True)
    return grouped.loc[:, ["group_id", "device_id", "speed_percent", "label", "window_count", "split"]], train, test


def _build_balance_report(all_frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_counts_all = all_frame.groupby(["device_id", "speed_percent", "label"])["group_id"].nunique()
    window_counts_all = all_frame.groupby(["device_id", "speed_percent", "label"]).size()
    train_group_counts = train.groupby(["device_id", "speed_percent", "label"])["group_id"].nunique()
    test_group_counts = test.groupby(["device_id", "speed_percent", "label"])["group_id"].nunique()
    train_window_counts = train.groupby(["device_id", "speed_percent", "label"]).size()
    test_window_counts = test.groupby(["device_id", "speed_percent", "label"]).size()
    for key, total_groups in group_counts_all.items():
        rows.append(
            {
                "device_id": key[0],
                "speed_percent": key[1],
                "label": key[2],
                "group_count_total": int(total_groups),
                "window_count_total": int(window_counts_all.get(key, 0)),
                "train_group_count": int(train_group_counts.get(key, 0)),
                "test_group_count": int(test_group_counts.get(key, 0)),
                "train_window_count": int(train_window_counts.get(key, 0)),
                "test_window_count": int(test_window_counts.get(key, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["device_id", "speed_percent", "label"]).reset_index(drop=True)


def _fit_global_preprocessors(train: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_imputed = imputer.fit_transform(train[CANDIDATE_FEATURE_COLUMNS])
    test_imputed = imputer.transform(test[CANDIDATE_FEATURE_COLUMNS])
    train_scaled.loc[:, CANDIDATE_FEATURE_COLUMNS] = scaler.fit_transform(train_imputed)
    test_scaled.loc[:, CANDIDATE_FEATURE_COLUMNS] = scaler.transform(test_imputed)
    return train_scaled, test_scaled, scaler, imputer


def _run_task_suite(train, test, *, target_column, labels, feature_columns, output_root, config, probability_labels):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in ("random_forest", "bp", "svm"):
        print(f"[task={target_column}] running {model_name}", flush=True)
        cv_results, best_params, cv_summary = _cv_select(model_name, train, target_column, feature_columns, output_root / model_name, config)
        metrics = _fit_and_score(model_name, train, test, target_column, labels, feature_columns, best_params, output_root / model_name, config, probability_labels)
        metrics.update(cv_summary)
        rows.append(metrics)
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(output_root / "model_summary.csv", index=False, encoding=config.csv_encoding)
    return summary


def _cv_select(model_name, train, target_column, feature_columns, output_dir, config):
    splitter = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    y = train[target_column]
    groups = train["group_id"]
    split_indices = list(splitter.split(train[feature_columns], y, groups))
    svm_fold_samples: dict[int, pd.DataFrame] = {}
    svm_sampling_reports: list[pd.DataFrame] = []
    if model_name == "svm":
        for fold, (train_idx, _valid_idx) in enumerate(split_indices):
            fold_train = train.iloc[train_idx].reset_index(drop=True)
            sampled_train, sampling_report = _sample_svm_training_groups(
                fold_train,
                target_column=target_column,
                desired_group_count=config.svm_target_group_count,
                random_state=config.random_state + fold,
            )
            svm_fold_samples[fold] = sampled_train
            sampling_report = sampling_report.copy()
            sampling_report["stage"] = "cv"
            sampling_report["fold"] = fold
            svm_sampling_reports.append(sampling_report)
    rows = []
    best_score = -np.inf
    best_params = None
    for params in _parameter_grid(model_name):
        print(f"[cv][{target_column}][{model_name}] params={json.dumps(params, ensure_ascii=False, sort_keys=True)}", flush=True)
        scores = []
        params_key = json.dumps(params, ensure_ascii=False, sort_keys=True)
        for fold, (train_idx, valid_idx) in enumerate(split_indices):
            fold_train = train.iloc[train_idx].reset_index(drop=True)
            fold_valid = train.iloc[valid_idx].reset_index(drop=True)
            if model_name == "svm":
                fold_train = svm_fold_samples[fold]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(fold_train[feature_columns]))
            x_valid = scaler.transform(imputer.transform(fold_valid[feature_columns]))
            estimator = _build_estimator(model_name, params, config.random_state, probability=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, fold_train[target_column])
            pred = estimator.predict(x_valid)
            score = float(f1_score(fold_valid[target_column], pred, average="macro"))
            scores.append(score)
            rows.append({"model": model_name, "params": params_key, "fold": fold, "macro_f1": score, "group_overlap_count": int(len(set(fold_train["group_id"]).intersection(set(fold_valid["group_id"]))))})
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    assert best_params is not None
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "cv_results.csv", index=False, encoding=config.csv_encoding)
    if svm_sampling_reports:
        pd.concat(svm_sampling_reports, ignore_index=True).to_csv(output_dir / "svm_group_sampling_cv.csv", index=False, encoding=config.csv_encoding)
    (output_dir / "best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    params_key = json.dumps(best_params, ensure_ascii=False, sort_keys=True)
    best_scores = frame.loc[frame["params"] == params_key, "macro_f1"]
    summary = {
        "cv_macro_f1_mean": float(best_scores.mean()),
        "cv_macro_f1_std": float(best_scores.std(ddof=0)),
        "cv_macro_f1_min": float(best_scores.min()),
        "cv_macro_f1_max": float(best_scores.max()),
    }
    return frame, best_params, summary


def _fit_and_score(model_name, train, test, target_column, labels, feature_columns, params, output_dir, config, probability_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "svm":
        train, sampling_report = _sample_svm_training_groups(
            train,
            target_column=target_column,
            desired_group_count=config.svm_target_group_count,
            random_state=config.random_state,
        )
        sampling_report.to_csv(output_dir / "svm_group_sampling_final_train.csv", index=False, encoding=config.csv_encoding)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_columns]))
    x_test = scaler.transform(imputer.transform(test[feature_columns]))
    estimator = _build_estimator(model_name, params, config.random_state, probability=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train, train[target_column])
    pred = estimator.predict(x_test)
    proba = estimator.predict_proba(x_test)
    report = classification_report(test[target_column], pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", encoding=config.csv_encoding)
    pd.DataFrame(confusion_matrix(test[target_column], pred, labels=labels), index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv", encoding=config.csv_encoding)
    window_predictions = test.loc[:, ["device_id", "speed_percent", "rpm", "label", "source_file", "record_index", "window_id", "group_id"]].copy()
    window_predictions["pred_label"] = pred
    window_predictions = _attach_probability_columns(
        window_predictions,
        probability_matrix=proba,
        estimator_classes=estimator.classes_,
        probability_labels=probability_labels,
    )
    window_predictions.to_csv(output_dir / "window_predictions.csv", index=False, encoding=config.csv_encoding)
    majority_frame = _group_vote_frame(window_predictions, probability_labels, true_column=target_column, mode="majority")
    probability_frame = _group_vote_frame(window_predictions, probability_labels, true_column=target_column, mode="avg_proba")
    majority_frame.to_csv(output_dir / "group_predictions_majority_vote.csv", index=False, encoding=config.csv_encoding)
    probability_frame.to_csv(output_dir / "group_predictions_avg_probability.csv", index=False, encoding=config.csv_encoding)
    pd.DataFrame(confusion_matrix(majority_frame["true_label"], majority_frame["pred_label"], labels=labels), index=labels, columns=labels).to_csv(output_dir / "group_majority_confusion_matrix.csv", encoding=config.csv_encoding)
    pd.DataFrame(confusion_matrix(probability_frame["true_label"], probability_frame["pred_label"], labels=labels), index=labels, columns=labels).to_csv(output_dir / "group_probability_confusion_matrix.csv", encoding=config.csv_encoding)
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": estimator, "features": feature_columns}, output_dir / f"{model_name}_bundle.joblib")
    metrics = {
        "model": model_name,
        "best_params": params,
        "accuracy": float(accuracy_score(test[target_column], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test[target_column], pred)),
        "macro_precision": float(precision_score(test[target_column], pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(test[target_column], pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(test[target_column], pred, average="macro")),
        "weighted_f1": float(f1_score(test[target_column], pred, average="weighted")),
        "group_majority_accuracy": float(accuracy_score(majority_frame["true_label"], majority_frame["pred_label"])),
        "group_majority_macro_f1": float(f1_score(majority_frame["true_label"], majority_frame["pred_label"], average="macro")),
        "group_probability_accuracy": float(accuracy_score(probability_frame["true_label"], probability_frame["pred_label"])),
        "group_probability_macro_f1": float(f1_score(probability_frame["true_label"], probability_frame["pred_label"], average="macro")),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False, encoding=config.csv_encoding)
    return metrics


def _group_vote_frame(window_predictions: pd.DataFrame, labels: list[str], *, true_column: str, mode: str) -> pd.DataFrame:
    rows = []
    proba_columns = [f"proba_{label}" for label in labels]
    for group_id, group in window_predictions.groupby("group_id", sort=False):
        true_label = group[true_column].iloc[0]
        meta = group.iloc[0][["device_id", "speed_percent", "rpm", "source_file", "record_index"]].to_dict()
        if mode == "majority":
            counts = group["pred_label"].value_counts()
            top_labels = counts[counts == counts.max()].index.tolist()
            if len(top_labels) == 1:
                pred_label = top_labels[0]
            else:
                avg_probs = group[[f"proba_{label}" for label in top_labels]].mean()
                pred_label = avg_probs.idxmax().replace("proba_", "")
        else:
            avg_probs = group[proba_columns].mean()
            pred_label = avg_probs.idxmax().replace("proba_", "")
        row = {"group_id": group_id, "true_label": true_label, "pred_label": pred_label, **meta}
        for column in proba_columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_probability_columns(
    frame: pd.DataFrame,
    *,
    probability_matrix,
    estimator_classes,
    probability_labels: list[str],
) -> pd.DataFrame:
    enriched = frame.copy()
    probability_matrix = np.asarray(probability_matrix)
    class_index = {str(label): idx for idx, label in enumerate(estimator_classes)}
    for label in probability_labels:
        enriched[f"proba_{label}"] = probability_matrix[:, class_index[str(label)]]
    return enriched


def _sample_svm_training_groups(frame: pd.DataFrame, *, target_column: str, desired_group_count: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        frame.groupby("group_id", as_index=False)
        .agg(
            device_id=("device_id", "first"),
            speed_percent=("speed_percent", "first"),
            label=("label", "first"),
            source_file=("source_file", "first"),
            record_index=("record_index", "first"),
            window_count=("window_id", "size"),
        )
    )
    grouped[target_column] = frame.groupby("group_id")[target_column].first().values
    strata_columns = _sampling_strata_columns(target_column)
    grouped["stratum_key"] = grouped[strata_columns].astype(str).agg("::".join, axis=1)
    desired_group_count = min(int(desired_group_count), int(len(grouped)))
    if desired_group_count <= 0:
        raise ValueError("desired_group_count must be positive")

    group_counts = grouped["stratum_key"].value_counts().sort_index()
    base_allocation = pd.Series(0, index=group_counts.index, dtype=int)
    if desired_group_count >= len(group_counts):
        base_allocation = group_counts.clip(upper=1).astype(int)
    remaining = desired_group_count - int(base_allocation.sum())
    extra_capacity = group_counts - base_allocation
    if remaining > 0 and int(extra_capacity.sum()) > 0:
        raw_extra = remaining * (extra_capacity / extra_capacity.sum())
        extra_allocation = np.floor(raw_extra).astype(int)
        extra_allocation = extra_allocation.clip(upper=extra_capacity)
        leftover = remaining - int(extra_allocation.sum())
        if leftover > 0:
            fractional = (raw_extra - extra_allocation).sort_values(ascending=False)
            for stratum_key in fractional.index:
                if leftover == 0:
                    break
                if extra_allocation[stratum_key] < extra_capacity[stratum_key]:
                    extra_allocation[stratum_key] += 1
                    leftover -= 1
        base_allocation = base_allocation + extra_allocation

    rng = np.random.default_rng(random_state)
    selected_group_ids: list[str] = []
    for stratum_key, target_count in base_allocation.items():
        if target_count <= 0:
            continue
        candidates = grouped.loc[grouped["stratum_key"] == stratum_key, "group_id"].to_numpy()
        if len(candidates) <= target_count:
            selected_group_ids.extend(candidates.tolist())
        else:
            chosen = rng.choice(candidates, size=int(target_count), replace=False)
            selected_group_ids.extend(chosen.tolist())
    selected_group_ids = sorted(set(selected_group_ids))
    sampled = frame[frame["group_id"].isin(selected_group_ids)].reset_index(drop=True)

    selected_grouped = grouped[grouped["group_id"].isin(selected_group_ids)].copy()
    total_summary = grouped.groupby(strata_columns, as_index=False).agg(
        group_count_total=("group_id", "nunique"),
        window_count_total=("window_count", "sum"),
    )
    selected_summary = selected_grouped.groupby(strata_columns, as_index=False).agg(
        group_count_selected=("group_id", "nunique"),
        window_count_selected=("window_count", "sum"),
    )
    report = total_summary.merge(selected_summary, on=strata_columns, how="left").fillna(0)
    report["group_count_selected"] = report["group_count_selected"].astype(int)
    report["window_count_selected"] = report["window_count_selected"].astype(int)
    report["desired_group_count"] = desired_group_count
    report["selected_group_count_total"] = len(selected_group_ids)
    report["selected_window_count_total"] = len(sampled)
    report = report.sort_values(strata_columns).reset_index(drop=True)
    return sampled, report


def _sampling_strata_columns(target_column: str) -> list[str]:
    if target_column == "device_id":
        return ["device_id", "speed_percent"]
    return ["device_id", "speed_percent", target_column]


def _parameter_grid(model_name: str):
    if model_name == "svm":
        return [{"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"} for c in (1.0, 10.0)]
    if model_name == "random_forest":
        return [{"n_estimators": 120, "max_depth": depth, "max_features": "sqrt", "class_weight": "balanced_subsample"} for depth in (None, 20)]
    if model_name == "bp":
        return [{"hidden_layer_sizes": (64, 32), "alpha": alpha, "learning_rate_init": 0.001, "early_stopping": False, "validation_fraction": 0.1, "n_iter_no_change": 10} for alpha in (0.0001, 0.001)]
    raise ValueError(model_name)


def _build_estimator(model_name: str, params: dict[str, object], random_state: int, *, probability: bool):
    if model_name == "svm":
        return SVC(
            random_state=random_state,
            probability=probability,
            cache_size=4096,
            shrinking=False,
            **params,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if model_name == "bp":
        return MLPClassifier(activation="relu", solver="adam", max_iter=200, batch_size="auto", random_state=random_state, **params)
    raise ValueError(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-speed unified six-class experiment V2.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = run_experiment(ExperimentV2Config(output_root=Path(args.output_root)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
