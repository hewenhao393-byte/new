from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pump_diagnosis.three_class_feature_table import FEATURE_COLUMNS


DEFAULT_RAW_FEATURES = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_100_1480rpm_三分类_通道4_特征表.csv"
)
DEFAULT_ASSIGNMENT = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_80_20_group_split_standardized/Motor2_record_split_assignment.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor2_group_cv_raw_protocol"
)

LABELS = ["正常", "松动", "轴承故障"]
METADATA_COLUMNS = ["source_file", "window_id", "window_start", "window_end", "label", "record_index", "group_id"]
_EPS = 1e-12


def add_record_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "record_index" in enriched.columns and "group_id" in enriched.columns:
        return enriched
    record_index = (
        enriched["window_id"]
        .astype(str)
        .str.rsplit("_", n=2)
        .str[1]
        .astype(int)
    )
    enriched["record_index"] = record_index
    enriched["group_id"] = enriched["source_file"].astype(str) + "::record_" + enriched["record_index"].astype(str)
    return enriched


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
    if train.empty or test.empty:
        raise ValueError("group split assignment did not produce both train and test rows")
    if not set(train["group_id"]).isdisjoint(set(test["group_id"])):
        raise ValueError("group leakage detected between train and test")
    return train, test


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
        fold_scores: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(splitter.split(train_frame[feature_columns], labels, groups)):
            fold_train = train_frame.iloc[fit_idx].reset_index(drop=True)
            fold_valid = train_frame.iloc[valid_idx].reset_index(drop=True)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(fold_train[feature_columns])
            x_valid = scaler.transform(fold_valid[feature_columns])
            y_train = fold_train["label"]
            y_valid = fold_valid["label"]
            estimator = _build_estimator(model_name, params, random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, y_train)
            pred = estimator.predict(x_valid)
            score = float(f1_score(y_valid, pred, average="macro"))
            fold_scores.append(score)
            cv_rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "fold": fold_index,
                    "macro_f1": score,
                    "train_rows": int(len(fold_train)),
                    "valid_rows": int(len(fold_valid)),
                    "group_overlap_count": int(len(set(fold_train["group_id"]).intersection(set(fold_valid["group_id"])))),
                }
            )
        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
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
    y_train = train_frame["label"]
    y_test = test_frame["label"]
    estimator = _build_estimator(model_name, params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": estimator, "features": feature_columns}, model_dir / f"{model_name}_bundle.joblib")
    pd.DataFrame(confusion_matrix(y_test, pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
        model_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)).transpose().to_csv(
        model_dir / "classification_report.csv",
        encoding="utf-8-sig",
    )
    metrics = {
        "model": model_name,
        "best_params": params,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
    }
    for label in LABELS:
        label_report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)[label]
        metrics[f"precision_{label}"] = float(label_report["precision"])
        metrics[f"recall_{label}"] = float(label_report["recall"])
        metrics[f"f1_{label}"] = float(label_report["f1-score"])
    (model_dir / "best_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(model_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def run_protocol(
    raw_features_path: Path = DEFAULT_RAW_FEATURES,
    assignment_path: Path = DEFAULT_ASSIGNMENT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_features_path)
    assignment = pd.read_csv(assignment_path)
    train, test = build_train_test_from_assignment(raw, assignment)
    train.to_csv(output_root / "Motor2_train_80_raw.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_root / "Motor2_test_20_raw.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for model_name in ("svm", "random_forest", "bp"):
        cv_dir = output_root / model_name
        cv_results, best_params = run_grouped_cv_for_model(
            model_name,
            train,
            FEATURE_COLUMNS,
            output_root=cv_dir,
        )
        metrics = fit_final_model(
            model_name,
            train,
            test,
            FEATURE_COLUMNS,
            best_params,
            output_root=output_root,
        )
        metrics["cv_macro_f1_mean"] = float(cv_results.loc[cv_results["params"] == json.dumps(best_params, ensure_ascii=False, sort_keys=True), "macro_f1"].mean())
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "model_summary.csv", index=False, encoding="utf-8-sig")
    split_summary = {
        "raw_features_path": str(raw_features_path),
        "assignment_path": str(assignment_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": int(train["group_id"].nunique()),
        "test_groups": int(test["group_id"].nunique()),
        "train_labels": train["label"].value_counts().to_dict(),
        "test_labels": test["label"].value_counts().to_dict(),
    }
    (output_root / "protocol_summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"split_summary": split_summary, "model_summary": summary.to_dict(orient="records")}


def _parameter_grid(model_name: str) -> list[dict[str, object]]:
    if model_name == "svm":
        return [{"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"} for c in (1.0, 10.0)]
    if model_name == "random_forest":
        return [
            {"n_estimators": n, "max_depth": d, "max_features": mf, "class_weight": "balanced_subsample"}
            for n in (120,)
            for d in (None, 20)
            for mf in ("sqrt",)
        ]
    if model_name == "bp":
        return [
            {
                "hidden_layer_sizes": h,
                "alpha": alpha,
                "learning_rate_init": lr,
                "early_stopping": False,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
            }
            for h in ((64, 32),)
            for alpha in (0.0001, 0.001)
            for lr in (0.001,)
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
    parser = argparse.ArgumentParser(description="Run Motor-2 grouped CV protocol from raw feature table.")
    parser.add_argument("--raw-features", default=str(DEFAULT_RAW_FEATURES))
    parser.add_argument("--assignment", default=str(DEFAULT_ASSIGNMENT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = run_protocol(Path(args.raw_features), Path(args.assignment), Path(args.output_root))
    print(json.dumps(result["split_summary"], ensure_ascii=False, indent=2))
    print(pd.DataFrame(result["model_summary"]).loc[:, ["model", "accuracy", "balanced_accuracy", "macro_f1", "cv_macro_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
