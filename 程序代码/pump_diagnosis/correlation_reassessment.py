from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import ModelingConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.model_training import train_and_evaluate_svm
from pump_diagnosis.modeling import audit_leakage, drop_nonfinite_feature_rows, fit_training_feature_filter


CRITICAL_LABELS = ("松动", "轴承故障", "联轴器不对中")


def _load_raw_feature_frames(feature_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(feature_root / "train_features_raw.csv")
    test = pd.read_csv(feature_root / "test_features_raw.csv")
    return train, test


def _load_manifests(feature_root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train_manifest = feature_root / "train_manifest.csv"
    test_manifest = feature_root / "test_manifest.csv"
    train = pd.read_csv(train_manifest) if train_manifest.exists() else None
    test = pd.read_csv(test_manifest) if test_manifest.exists() else None
    return train, test


def _feature_frame(features: list[str], *, threshold: float, stage: str, protected: set[str] | None = None) -> pd.DataFrame:
    protected = protected or set()
    return pd.DataFrame(
        {
            "threshold": threshold,
            "stage": stage,
            "rank": range(1, len(features) + 1),
            "feature": features,
            "protected": [feature in protected for feature in features],
        }
    )


def _build_leakage_check(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_manifest: pd.DataFrame | None = None,
    test_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(field: str, train_values: pd.Series, test_values: pd.Series, note: str) -> None:
        train_set = set(train_values.dropna().astype(str))
        test_set = set(test_values.dropna().astype(str))
        overlap = sorted(train_set.intersection(test_set))
        rows.append(
            {
                "check": field,
                "train_unique": int(len(train_set)),
                "test_unique": int(len(test_set)),
                "overlap_count": int(len(overlap)),
                "passed": len(overlap) == 0,
                "note": note,
                "overlap_examples": json.dumps(overlap[:10], ensure_ascii=False),
            }
        )

    add_row("source_file", train["file_path"], test["file_path"], "训练集和测试集原始文件路径是否有交集")
    if "run_id" in train.columns and "run_id" in test.columns:
        add_row("run_id", train["run_id"], test["run_id"], "训练集和测试集采集批次/运行编号是否有交集")
    if "sample_id" in train.columns and "sample_id" in test.columns:
        add_row("sample_id", train["sample_id"], test["sample_id"], "训练集和测试集样本编号是否有交集")
    if "window_index" in train.columns and "run_id" in train.columns:
        add_row(
            "window_source",
            train["run_id"].astype(str) + "::" + train["window_index"].astype(str),
            test["run_id"].astype(str) + "::" + test["window_index"].astype(str),
            "相同原始文件内的滑窗位置是否跨集合重复",
        )
    if train_manifest is not None and test_manifest is not None:
        if "condition_id" in train_manifest.columns and "condition_id" in test_manifest.columns:
            add_row(
                "condition_id",
                train_manifest["condition_id"],
                test_manifest["condition_id"],
                "采集批次/工况编号是否跨集合重复",
            )
        if "sha256" in train_manifest.columns and "sha256" in test_manifest.columns:
            add_row("sha256", train_manifest["sha256"], test_manifest["sha256"], "原始文件哈希是否跨集合重复")

    rows.append(
        {
            "check": "split_before_windowing",
            "train_unique": 1,
            "test_unique": 1,
            "overlap_count": 0,
            "passed": bool(
                all(
                    row["passed"]
                    for row in rows
                    if row["check"] in {"source_file", "run_id", "sample_id", "window_source", "condition_id", "sha256"}
                )
            ),
            "note": "若原始文件、运行编号、样本编号、窗口源都不重叠，则说明当前切分不是按滑窗随机划分",
            "overlap_examples": "[]",
        }
    )
    return pd.DataFrame(rows)


def run_correlation_reassessment(config: ModelingConfig) -> dict[str, object]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    train, test = _load_raw_feature_frames(config.feature_root)
    train_manifest, test_manifest = _load_manifests(config.feature_root)

    leakage_report = audit_leakage(train, test)
    leakage_check = _build_leakage_check(train, test, train_manifest, test_manifest)
    leakage_check.to_csv(config.output_root / "data_leakage_check.csv", index=False)

    train, test, nonfinite_report = drop_nonfinite_feature_rows(train, test, FEATURE_COLUMNS)

    all_features = list(FEATURE_COLUMNS)
    pd.DataFrame({"rank": range(1, len(all_features) + 1), "feature": all_features}).to_csv(
        config.output_root / "all_valid_features.csv",
        index=False,
    )

    feature_sets: dict[str, list[str]] = {"all_valid": all_features}
    filter_outputs: dict[str, pd.DataFrame] = {}
    threshold_map = {"corr_095": 0.95, "corr_090": 0.90}

    for name, threshold in threshold_map.items():
        threshold_config = replace(config, correlation_threshold=threshold)
        result = fit_training_feature_filter(
            train[FEATURE_COLUMNS],
            FEATURE_COLUMNS,
            threshold_config,
        )
        feature_sets[name] = result.kept_features
        filter_outputs[name] = pd.DataFrame(
            [
                {
                    "threshold": threshold,
                    "feature": feature,
                    "reason": reason,
                    "protected": feature in set(config.protected_feature_columns),
                }
                for feature, reason in result.removed_reasons.items()
            ]
        )
        _feature_frame(list(FEATURE_COLUMNS), threshold=threshold, stage="before", protected=set(config.protected_feature_columns)).to_csv(
            config.output_root / f"{name}_before.csv",
            index=False,
        )
        _feature_frame(result.kept_features, threshold=threshold, stage="after", protected=set(config.protected_feature_columns)).to_csv(
            config.output_root / f"{name}_after.csv",
            index=False,
        )
        if result.retained_high_correlation_pairs:
            pd.DataFrame(
                [
                    {"threshold": threshold, "left": left, "right": right, "correlation": corr}
                    for left, right, corr in result.retained_high_correlation_pairs
                ]
            ).to_csv(config.output_root / f"{name}_retained_high_corr_pairs.csv", index=False)

    comparison_rows: list[pd.DataFrame] = []
    for feature_set_name, selected_features in feature_sets.items():
        feature_dir = config.output_root / feature_set_name
        set_config = replace(config, output_root=feature_dir)
        metrics = train_and_evaluate_svm(train, test, selected_features, set_config)
        metrics.insert(0, "feature_set", feature_set_name)
        metrics.insert(1, "feature_count", len(selected_features))
        metrics.insert(2, "protected_feature_count", sum(feature in set(config.protected_feature_columns) for feature in selected_features))
        comparison_rows.append(metrics)

        pd.DataFrame(
            {"rank": range(1, len(selected_features) + 1), "feature": selected_features}
        ).to_csv(feature_dir / "selected_features.csv", index=False)

    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(config.output_root / "model_comparison.csv", index=False)

    critical_recall_cols = [f"recall_{label}" for label in CRITICAL_LABELS if f"recall_{label}" in comparison.columns]
    recommendation = (
        comparison.assign(
            critical_recall_mean=lambda frame: frame[critical_recall_cols].mean(axis=1) if critical_recall_cols else 0.0
        )
        .sort_values(
            ["macro_f1", "critical_recall_mean", "balanced_accuracy", "feature_count"],
            ascending=[False, False, False, True],
        )
        .iloc[0]
        .to_dict()
    )

    summary = {
        "leakage_check": leakage_report,
        "nonfinite_report": nonfinite_report,
        "selected_feature_sets": {name: len(features) for name, features in feature_sets.items()},
        "recommendation": recommendation,
    }
    (config.output_root / "run_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, frame in filter_outputs.items():
        if not frame.empty:
            frame.to_csv(config.output_root / f"{name}_removed_features.csv", index=False)

    return summary
