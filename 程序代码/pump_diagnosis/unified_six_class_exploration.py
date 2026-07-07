from __future__ import annotations

"""Legacy unified six-class exploration script.

This file is kept for historical comparison only. Its label order must not be
used as the formal software output basis. The formal software contract is
frozen by the V2 inference contract.
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pump_diagnosis.four_class_feature_table import FourClassFeatureConfig
from pump_diagnosis.motor2_group_cv import add_record_group_columns as add_motor2_group_columns
from pump_diagnosis.three_class_feature_table import ThreeClassFeatureConfig


MOTOR2_FEATURE_TABLE = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_100_1480rpm_三分类_通道4_特征表.csv"
)
MOTOR4_FEATURE_TABLE = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor4_70_2070rpm_四分类_通道4_特征表.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/统一六分类探索实验"
)

SIX_CLASS_LABELS = ["正常", "松动", "轴承故障", "转子不平衡", "联轴器不对中", "汽蚀"]
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
REQUIRED_METADATA_COLUMNS = [
    "source_file",
    "window_id",
    "record_index",
    "group_id",
    "device_id",
    "label",
]


@dataclass(frozen=True)
class ExplorationConfig:
    motor2_feature_table: Path = MOTOR2_FEATURE_TABLE
    motor4_feature_table: Path = MOTOR4_FEATURE_TABLE
    output_root: Path = DEFAULT_OUTPUT_ROOT
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    csv_encoding: str = "utf-8-sig"


def add_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_motor2_group_columns(frame)
    enriched["group_id"] = (
        enriched["device_id"].astype(str)
        + "::"
        + enriched["source_file"].astype(str)
        + "::record_"
        + enriched["record_index"].astype(str)
    )
    return enriched


def build_consistency_report(
    *,
    motor2_columns: list[str],
    motor4_columns: list[str],
    motor2_summary: dict[str, object],
    motor4_summary: dict[str, object],
    motor2_config: dict[str, object],
    motor4_config: dict[str, object],
) -> dict[str, object]:
    comparable_summary_keys = ["processed_fs", "window_size", "step_size", "wavelet"]
    comparable_config_keys = [
        "bandpass_low",
        "bandpass_high",
        "resample_up",
        "resample_down",
        "envelope_band_low",
        "envelope_band_high",
        "wavelet_level",
    ]
    column_match = motor2_columns == motor4_columns
    summary_match = all(motor2_summary.get(key) == motor4_summary.get(key) for key in comparable_summary_keys)
    config_match = all(motor2_config.get(key) == motor4_config.get(key) for key in comparable_config_keys)
    return {
        "column_match": bool(column_match),
        "summary_match": bool(summary_match),
        "processing_match": bool(config_match),
        "motor2_summary": motor2_summary,
        "motor4_summary": motor4_summary,
        "motor2_config": motor2_config,
        "motor4_config": motor4_config,
        "is_consistent": bool(column_match and summary_match and config_match),
    }


def build_unified_dataset(motor2_frame: pd.DataFrame, motor4_frame: pd.DataFrame) -> pd.DataFrame:
    motor2 = motor2_frame.copy()
    motor4 = motor4_frame.copy()
    motor2["device_id"] = "Motor-2"
    motor4["device_id"] = "Motor-4"
    combined = pd.concat([motor2, motor4], ignore_index=True)
    combined = add_group_columns(combined)
    keep_columns = ["source_file", "window_id", "window_start", "window_end", "record_index", "group_id", "device_id", "label"] + CANDIDATE_FEATURE_COLUMNS
    return combined.loc[:, keep_columns]


def run_exploration(config: ExplorationConfig) -> dict[str, object]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    motor2 = pd.read_csv(config.motor2_feature_table)
    motor4 = pd.read_csv(config.motor4_feature_table)
    consistency = _load_and_check_consistency(motor2, motor4)
    pd.DataFrame([consistency]).to_csv(config.output_root / "consistency_report.csv", index=False, encoding=config.csv_encoding)
    if not consistency["is_consistent"]:
        raise ValueError("Motor-2 and Motor-4 feature tables are not consistent; stopping per requirement")

    unified = build_unified_dataset(motor2, motor4)
    unified.to_csv(config.output_root / "unified_six_class_dataset_raw.csv", index=False, encoding=config.csv_encoding)

    train, test, split_report = _grouped_train_test_split(unified, config)
    train.to_csv(config.output_root / "train_80_raw.csv", index=False, encoding=config.csv_encoding)
    test.to_csv(config.output_root / "test_20_raw.csv", index=False, encoding=config.csv_encoding)
    pd.DataFrame([split_report]).to_csv(config.output_root / "group_leakage_check.csv", index=False, encoding=config.csv_encoding)

    scaler, train_scaled, test_scaled = _fit_global_scaler(train, test)
    joblib.dump(scaler, config.output_root / "scaler.joblib")
    train_scaled.to_csv(config.output_root / "train_80_scaled.csv", index=False, encoding=config.csv_encoding)
    test_scaled.to_csv(config.output_root / "test_20_scaled.csv", index=False, encoding=config.csv_encoding)

    six_class_results = _run_model_suite(
        train,
        test,
        target_column="label",
        labels=SIX_CLASS_LABELS,
        feature_columns=CANDIDATE_FEATURE_COLUMNS,
        output_root=config.output_root / "six_class_models",
        config=config,
    )
    device_results = _run_model_suite(
        train,
        test,
        target_column="device_id",
        labels=DEVICE_LABELS,
        feature_columns=CANDIDATE_FEATURE_COLUMNS,
        output_root=config.output_root / "device_models",
        config=config,
    )
    summary_note = {
        "experiment_name": "统一六分类探索实验",
        "warning": "该实验仅为统一六分类探索实验，不得表述为已证明适用于未知水泵。",
    }
    (config.output_root / "experiment_note.json").write_text(json.dumps(summary_note, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "consistency_report": consistency,
        "split_report": split_report,
        "six_class_summary": six_class_results.to_dict(orient="records"),
        "device_summary": device_results.to_dict(orient="records"),
    }


def _load_and_check_consistency(motor2: pd.DataFrame, motor4: pd.DataFrame) -> dict[str, object]:
    motor2_summary = _read_optional_summary(MOTOR2_FEATURE_TABLE.with_suffix(".summary.json"))
    motor4_summary = _read_optional_summary(MOTOR4_FEATURE_TABLE.with_suffix(".summary.json"))
    motor2_cfg = ThreeClassFeatureConfig()
    motor4_cfg = FourClassFeatureConfig()
    return build_consistency_report(
        motor2_columns=list(motor2.columns),
        motor4_columns=list(motor4.columns),
        motor2_summary=motor2_summary,
        motor4_summary=motor4_summary,
        motor2_config={
            "bandpass_low": motor2_cfg.bandpass_low,
            "bandpass_high": motor2_cfg.bandpass_high,
            "resample_up": motor2_cfg.resample_up,
            "resample_down": motor2_cfg.resample_down,
            "envelope_band_low": motor2_cfg.envelope_band_low,
            "envelope_band_high": motor2_cfg.envelope_band_high,
            "wavelet_level": motor2_cfg.wavelet_level,
        },
        motor4_config={
            "bandpass_low": motor4_cfg.bandpass_low,
            "bandpass_high": motor4_cfg.bandpass_high,
            "resample_up": motor4_cfg.resample_up,
            "resample_down": motor4_cfg.resample_down,
            "envelope_band_low": motor4_cfg.envelope_band_low,
            "envelope_band_high": motor4_cfg.envelope_band_high,
            "wavelet_level": motor4_cfg.wavelet_level,
        },
    )


def _read_optional_summary(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "processed_fs": 12000,
        "window_size": 2400,
        "step_size": 1200,
        "wavelet": "db6",
        "summary_missing_assumed_from_script": True,
    }


def _grouped_train_test_split(frame: pd.DataFrame, config: ExplorationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    grouped = (
        frame.groupby("group_id", as_index=False)
        .agg(label=("label", "first"), device_id=("device_id", "first"), window_count=("window_id", "size"))
    )
    grouped["stratify_key"] = grouped["device_id"] + "::" + grouped["label"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    train_idx, test_idx = next(splitter.split(grouped["group_id"], grouped["stratify_key"]))
    train_groups = set(grouped.iloc[train_idx]["group_id"])
    test_groups = set(grouped.iloc[test_idx]["group_id"])
    train = frame[frame["group_id"].isin(train_groups)].reset_index(drop=True)
    test = frame[frame["group_id"].isin(test_groups)].reset_index(drop=True)
    report = {
        "group_overlap_count": int(len(train_groups.intersection(test_groups))),
        "window_overlap_count": int(len(set(train["window_id"]).intersection(set(test["window_id"])))),
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_label_distribution": json.dumps(train["label"].value_counts().to_dict(), ensure_ascii=False),
        "test_label_distribution": json.dumps(test["label"].value_counts().to_dict(), ensure_ascii=False),
        "train_device_distribution": json.dumps(train["device_id"].value_counts().to_dict(), ensure_ascii=False),
        "test_device_distribution": json.dumps(test["device_id"].value_counts().to_dict(), ensure_ascii=False),
    }
    return train, test, report


def _fit_global_scaler(train: pd.DataFrame, test: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    scaler.fit(train[CANDIDATE_FEATURE_COLUMNS])
    train_scaled.loc[:, CANDIDATE_FEATURE_COLUMNS] = scaler.transform(train[CANDIDATE_FEATURE_COLUMNS])
    test_scaled.loc[:, CANDIDATE_FEATURE_COLUMNS] = scaler.transform(test[CANDIDATE_FEATURE_COLUMNS])
    return scaler, train_scaled, test_scaled


def _run_model_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    labels: list[str],
    feature_columns: list[str],
    output_root: Path,
    config: ExplorationConfig,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for model_name in ("svm", "random_forest", "bp"):
        cv_results, best_params = _run_grouped_cv(
            model_name,
            train,
            target_column,
            feature_columns,
            output_root / model_name,
            config,
        )
        metrics = _fit_and_evaluate(
            model_name,
            train,
            test,
            target_column,
            labels,
            feature_columns,
            best_params,
            output_root / model_name,
            config,
        )
        metrics["cv_macro_f1_mean"] = float(
            cv_results.loc[
                cv_results["params"] == json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                "macro_f1",
            ].mean()
        )
        summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(output_root / "model_summary.csv", index=False, encoding=config.csv_encoding)
    return summary


def _run_grouped_cv(
    model_name: str,
    train: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    output_root: Path,
    config: ExplorationConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    splitter = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    y = train[target_column]
    groups = train["group_id"]
    rows: list[dict[str, object]] = []
    best_score = -np.inf
    best_params: dict[str, object] | None = None
    for params in _parameter_grid(model_name):
        fold_scores: list[float] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(train[feature_columns], y, groups)):
            fold_train = train.iloc[train_idx].reset_index(drop=True)
            fold_valid = train.iloc[valid_idx].reset_index(drop=True)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(fold_train[feature_columns])
            x_valid = scaler.transform(fold_valid[feature_columns])
            estimator = _build_estimator(model_name, params, config.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, fold_train[target_column])
            pred = estimator.predict(x_valid)
            score = float(f1_score(fold_valid[target_column], pred, average="macro"))
            fold_scores.append(score)
            rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "fold": fold_index,
                    "macro_f1": score,
                    "group_overlap_count": int(len(set(fold_train["group_id"]).intersection(set(fold_valid["group_id"])))),
                }
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    assert best_params is not None
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "cv_results.csv", index=False, encoding=config.csv_encoding)
    (output_root / "best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, best_params


def _fit_and_evaluate(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    labels: list[str],
    feature_columns: list[str],
    params: dict[str, object],
    output_root: Path,
    config: ExplorationConfig,
) -> dict[str, object]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])
    estimator = _build_estimator(model_name, params, config.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train, train[target_column])
    pred = estimator.predict(x_test)
    report = classification_report(test[target_column], pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_root / "classification_report.csv", encoding=config.csv_encoding)
    pd.DataFrame(confusion_matrix(test[target_column], pred, labels=labels), index=labels, columns=labels).to_csv(
        output_root / "confusion_matrix.csv",
        encoding=config.csv_encoding,
    )
    joblib.dump({"scaler": scaler, "model": estimator, "features": feature_columns}, output_root / f"{model_name}_bundle.joblib")
    metrics = {
        "model": model_name,
        "best_params": params,
        "accuracy": float(accuracy_score(test[target_column], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test[target_column], pred)),
        "macro_f1": float(f1_score(test[target_column], pred, average="macro")),
        "weighted_f1": float(f1_score(test[target_column], pred, average="weighted")),
    }
    for label in labels:
        metrics[f"precision_{label}"] = float(report[label]["precision"])
        metrics[f"recall_{label}"] = float(report[label]["recall"])
        metrics[f"f1_{label}"] = float(report[label]["f1-score"])
    pd.DataFrame([metrics]).to_csv(output_root / "test_metrics.csv", index=False, encoding=config.csv_encoding)
    return metrics


def _parameter_grid(model_name: str) -> list[dict[str, object]]:
    if model_name == "svm":
        return [{"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"} for c in (1.0, 10.0)]
    if model_name == "random_forest":
        return [
            {"n_estimators": 120, "max_depth": depth, "max_features": "sqrt", "class_weight": "balanced_subsample"}
            for depth in (None, 20)
        ]
    if model_name == "bp":
        return [
            {
                "hidden_layer_sizes": (64, 32),
                "alpha": alpha,
                "learning_rate_init": 0.001,
                "early_stopping": False,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
            }
            for alpha in (0.0001, 0.001)
        ]
    raise ValueError(f"unknown model {model_name}")


def _build_estimator(model_name: str, params: dict[str, object], random_state: int):
    if model_name == "svm":
        return SVC(random_state=random_state, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if model_name == "bp":
        return MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=200,
            batch_size="auto",
            random_state=random_state,
            **params,
        )
    raise ValueError(f"unknown model {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified six-class exploration experiment.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = run_exploration(ExplorationConfig(output_root=Path(args.output_root)))
    print(json.dumps(result["split_report"], ensure_ascii=False, indent=2))
    print("统一六分类探索实验：仅作探索，不证明适用于未知水泵。")


if __name__ == "__main__":
    main()
