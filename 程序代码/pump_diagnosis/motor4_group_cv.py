from __future__ import annotations

import argparse
import json
import warnings
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

from pump_diagnosis.three_class_feature_table import FEATURE_COLUMNS


DEFAULT_RAW_FEATURES = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor4_70_2070rpm_四分类_通道4_特征表.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor4_80_20_group_split_standardized"
)

LABELS = ["正常", "转子不平衡", "联轴器不对中", "汽蚀"]


def add_record_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "record_index" in enriched.columns and "group_id" in enriched.columns:
        return enriched
    record_index = (
        enriched["window_id"].astype(str).str.rsplit("_", n=2).str[1].astype(int)
    )
    enriched["record_index"] = record_index
    enriched["group_id"] = enriched["source_file"].astype(str) + "::record_" + enriched["record_index"].astype(str)
    return enriched


def build_group_assignment(
    raw_features: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    enriched = add_record_group_columns(raw_features)
    grouped = (
        enriched.groupby("group_id", as_index=False)
        .agg(label=("label", "first"), window_count=("window_id", "size"))
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(grouped["group_id"], grouped["label"]))
    grouped["split"] = "train"
    grouped.loc[test_idx, "split"] = "test"
    return grouped.loc[:, ["group_id", "label", "window_count", "split"]]


def build_train_test_from_assignment(
    raw_features: pd.DataFrame,
    assignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = add_record_group_columns(raw_features)
    merged = enriched.merge(
        assignment.loc[:, ["group_id", "split"]],
        on="group_id",
        how="inner",
        validate="many_to_one",
    )
    train = merged[merged["split"] == "train"].drop(columns="split").reset_index(drop=True)
    test = merged[merged["split"] == "test"].drop(columns="split").reset_index(drop=True)
    if not set(train["group_id"]).isdisjoint(set(test["group_id"])):
        raise ValueError("group leakage detected between train and test")
    return train, test


def fit_scaler_and_export(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    output_root: Path,
) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_scaled = train_frame.copy()
    test_scaled = test_frame.copy()
    scaler.fit(train_frame[feature_columns])
    train_scaled.loc[:, feature_columns] = scaler.transform(train_frame[feature_columns])
    test_scaled.loc[:, feature_columns] = scaler.transform(test_frame[feature_columns])
    joblib.dump(scaler, output_root / "Motor4_scaler.joblib")
    pd.DataFrame({"feature": feature_columns, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(
        output_root / "Motor4_scaler_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    train_scaled.to_csv(output_root / "Motor4_train_80_scaled.csv", index=False, encoding="utf-8-sig")
    test_scaled.to_csv(output_root / "Motor4_test_20_scaled.csv", index=False, encoding="utf-8-sig")
    return scaler, train_scaled, test_scaled


def run_grouped_cv_for_model(
    model_name: str,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    output_root: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = train_frame["label"]
    groups = train_frame["group_id"]
    cv_rows: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_score = -np.inf

    for params in _parameter_grid(model_name):
        print(f"[cv] {model_name} params={json.dumps(params, ensure_ascii=False, sort_keys=True)}", flush=True)
        scores: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(splitter.split(train_frame[feature_columns], labels, groups)):
            fold_train = train_frame.iloc[fit_idx].reset_index(drop=True)
            fold_valid = train_frame.iloc[valid_idx].reset_index(drop=True)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(fold_train[feature_columns])
            x_valid = scaler.transform(fold_valid[feature_columns])
            estimator = _build_estimator(model_name, params, random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, fold_train["label"])
            pred = estimator.predict(x_valid)
            score = float(f1_score(fold_valid["label"], pred, average="macro"))
            scores.append(score)
            cv_rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "fold": fold_index,
                    "macro_f1": score,
                    "group_overlap_count": int(len(set(fold_train["group_id"]).intersection(set(fold_valid["group_id"])))),
                }
            )
        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    result = pd.DataFrame(cv_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / f"{model_name}_cv_results.csv", index=False, encoding="utf-8-sig")
    return result, best_params


def fit_final_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, object],
    *,
    output_root: Path,
    random_state: int = 42,
) -> dict[str, object]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[feature_columns])
    x_test = scaler.transform(test_frame[feature_columns])
    estimator = _build_estimator(model_name, params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train, train_frame["label"])
    pred = estimator.predict(x_test)
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": estimator, "features": feature_columns}, model_dir / f"{model_name}_bundle.joblib")
    pd.DataFrame(confusion_matrix(test_frame["label"], pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
        model_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    report = classification_report(test_frame["label"], pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(model_dir / "classification_report.csv", encoding="utf-8-sig")
    metrics = {
        "model": model_name,
        "best_params": params,
        "accuracy": float(accuracy_score(test_frame["label"], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_frame["label"], pred)),
        "macro_f1": float(f1_score(test_frame["label"], pred, average="macro")),
        "weighted_f1": float(f1_score(test_frame["label"], pred, average="weighted")),
    }
    for label in LABELS:
        metrics[f"precision_{label}"] = float(report[label]["precision"])
        metrics[f"recall_{label}"] = float(report[label]["recall"])
        metrics[f"f1_{label}"] = float(report[label]["f1-score"])
    (model_dir / "best_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(model_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def run_protocol(
    raw_features_path: Path = DEFAULT_RAW_FEATURES,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_features_path)
    assignment = build_group_assignment(raw)
    assignment.to_csv(output_root / "Motor4_record_split_assignment.csv", index=False, encoding="utf-8-sig")
    train, test = build_train_test_from_assignment(raw, assignment)
    train.to_csv(output_root / "Motor4_train_80_raw.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_root / "Motor4_test_20_raw.csv", index=False, encoding="utf-8-sig")
    fit_scaler_and_export(train, test, FEATURE_COLUMNS, output_root=output_root)

    summary_rows: list[dict[str, object]] = []
    for model_name in ("svm", "random_forest", "bp"):
        cv_results, best_params = run_grouped_cv_for_model(
            model_name,
            train,
            FEATURE_COLUMNS,
            output_root=output_root / model_name,
        )
        metrics = fit_final_model(
            model_name,
            train,
            test,
            FEATURE_COLUMNS,
            best_params,
            output_root=output_root,
        )
        params_key = json.dumps(best_params, ensure_ascii=False, sort_keys=True)
        metrics["cv_macro_f1_mean"] = float(cv_results.loc[cv_results["params"] == params_key, "macro_f1"].mean())
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(output_root / "model_summary.csv", index=False, encoding="utf-8-sig")
    split_summary = {
        "raw_features_path": str(raw_features_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": int(train["group_id"].nunique()),
        "test_groups": int(test["group_id"].nunique()),
        "train_labels": train["label"].value_counts().to_dict(),
        "test_labels": test["label"].value_counts().to_dict(),
    }
    (output_root / "split_summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"split_summary": split_summary, "model_summary": summary.to_dict(orient="records")}


def _parameter_grid(model_name: str) -> list[dict[str, object]]:
    if model_name == "svm":
        return [{"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"} for c in (1.0, 10.0)]
    if model_name == "random_forest":
        return [
            {"n_estimators": 120, "max_depth": d, "max_features": "sqrt", "class_weight": "balanced_subsample"}
            for d in (None, 20)
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
    parser = argparse.ArgumentParser(description="Run Motor-4 grouped CV protocol from raw feature table.")
    parser.add_argument("--raw-features", default=str(DEFAULT_RAW_FEATURES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = run_protocol(Path(args.raw_features), Path(args.output_root))
    print(json.dumps(result["split_summary"], ensure_ascii=False, indent=2))
    print(pd.DataFrame(result["model_summary"]).loc[:, ["model", "accuracy", "balanced_accuracy", "macro_f1", "cv_macro_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
