from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from config import ModelingConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.modeling import (
    LeakageError,
    audit_leakage,
    drop_nonfinite_feature_rows,
    evaluate_top_k_grouped,
    fit_training_feature_filter,
)
from pump_diagnosis.model_training import train_and_evaluate_models
from pump_diagnosis.correlation_reassessment import run_correlation_reassessment
from pump_diagnosis.modeling_runner import run_full_modeling


def _metadata_row(prefix: str, file_index: int, run_index: int, window_index: int, label: str) -> dict[str, object]:
    file_path = f"/{prefix}/class_{label}/file_{file_index}.csv"
    run_id = f"{prefix}_{label}_file_{file_index}_run_{run_index}"
    return {
        "sample_id": f"{run_id}_window_{window_index}",
        "label": label,
        "run_id": run_id,
        "file_path": file_path,
        "machine_id": "Motor-4",
        "condition_id": f"condition_{file_index}",
        "rpm": 2070.0,
        "channel": 4,
        "window_index": window_index,
        "window_start": window_index * 0.1,
        "window_end": window_index * 0.1 + 0.2,
        "original_fs": 20_000,
        "processed_fs": 12_000,
    }


def test_leakage_audit_rejects_overlapping_file_run_sample_and_window_source() -> None:
    row = _metadata_row("train", 0, 0, 0, "正常")
    train = pd.DataFrame([row])
    test = pd.DataFrame([row])

    with pytest.raises(LeakageError) as exc_info:
        audit_leakage(train, test)

    report = exc_info.value.report
    assert report["file_path_overlap_count"] == 1
    assert report["run_id_overlap_count"] == 1
    assert report["sample_id_overlap_count"] == 1
    assert report["window_source_overlap_count"] == 1


def test_training_filter_keeps_rms_and_does_not_use_test_variance() -> None:
    rng = np.random.default_rng(42)
    rms = np.linspace(1.0, 10.0, 100)
    train = pd.DataFrame(
        {
            "rms": rms,
            "std": rms * 1.001,
            "variance": rms**2,
            "constant_feature": np.ones(100),
            "near_zero_feature": np.full(100, 2.0) + rng.normal(0, 1e-8, 100),
            "independent_feature": rng.normal(size=100),
        }
    )
    test = train.copy()
    test["constant_feature"] = np.arange(100)
    test["near_zero_feature"] = np.arange(100) * 100
    config = ModelingConfig(near_zero_variance_threshold=1e-12, correlation_threshold=0.95)

    result = fit_training_feature_filter(train, list(train.columns), config)

    assert "rms" in result.kept_features
    assert "std" not in result.kept_features
    assert "variance" not in result.kept_features
    assert result.removed_reasons["constant_feature"] == "constant"
    assert result.removed_reasons["near_zero_feature"] == "near_zero_variance"
    assert "independent_feature" in result.kept_features


def test_training_filter_keeps_protected_physical_feature_under_high_correlation() -> None:
    rng = np.random.default_rng(123)
    base = np.linspace(0.0, 1.0, 200)
    train = pd.DataFrame(
        {
            "a_amp_1x_copy": base + rng.normal(0, 1e-4, size=base.size),
            "amp_1x": base + rng.normal(0, 1e-4, size=base.size),
            "independent_feature": rng.normal(size=base.size),
        }
    )
    config = ModelingConfig(near_zero_variance_threshold=1e-12, correlation_threshold=0.95)

    result = fit_training_feature_filter(train, list(train.columns), config)

    assert "amp_1x" in result.kept_features
    assert "a_amp_1x_copy" in result.removed_reasons
    assert result.removed_reasons["a_amp_1x_copy"] == "correlated"


def test_nonfinite_feature_rows_are_removed_and_reported() -> None:
    train = pd.DataFrame({"label": ["正常", "松动", "汽蚀"], "f1": [1.0, np.nan, 3.0], "f2": [1.0, 2.0, np.inf]})
    test = pd.DataFrame({"label": ["正常", "松动"], "f1": [1.0, 2.0], "f2": [-np.inf, 2.0]})

    clean_train, clean_test, report = drop_nonfinite_feature_rows(train, test, ["f1", "f2"])

    assert clean_train["label"].tolist() == ["正常"]
    assert clean_test["label"].tolist() == ["松动"]
    assert report == {"train_removed_rows": 2, "test_removed_rows": 1}


def _grouped_classification_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
    rows = []
    for label_index, label in enumerate(labels):
        for file_index in range(5):
            for window_index in range(4):
                row = _metadata_row("train", label_index * 10 + file_index, 0, window_index, label)
                row.update(
                    {
                        "feature_signal": label_index + rng.normal(0, 0.05),
                        "feature_signal_copy": label_index + rng.normal(0, 0.05),
                        "feature_noise_1": rng.normal(),
                        "feature_noise_2": rng.normal(),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_top_k_selection_uses_file_grouped_folds_without_overlap(tmp_path: Path) -> None:
    frame = _grouped_classification_frame()
    config = ModelingConfig(
        output_root=tmp_path,
        top_k_candidates=(2, 3),
        cv_folds=5,
        rf_n_estimators=10,
        rf_n_jobs=1,
    )

    result = evaluate_top_k_grouped(
        frame,
        ["feature_signal", "feature_signal_copy", "feature_noise_1", "feature_noise_2"],
        config,
    )

    assert result.selected_k in {2, 3}
    assert set(result.summary["k"]) == {2, 3}
    assert len(result.fold_results) == 10
    assert result.fold_results["group_overlap_count"].eq(0).all()
    assert result.fold_results.groupby("fold")["validation_file_count"].first().gt(0).all()


def test_three_models_write_metrics_confusions_and_use_requested_sample_size(tmp_path: Path) -> None:
    frame = _grouped_classification_frame()
    train = pd.concat([frame] * 3, ignore_index=True)
    test = frame.copy()
    selected = ["feature_signal", "feature_noise_1", "feature_noise_2"]
    config = ModelingConfig(
        output_root=tmp_path,
        rf_n_estimators=10,
        rf_n_jobs=1,
        svm_sample_size=60,
        mlp_max_iter=20,
        mlp_n_iter_no_change=5,
        plot_dpi=80,
    )

    summary = train_and_evaluate_models(train, test, selected, config)

    assert set(summary["model"]) == {"RandomForest", "SVM", "MLP"}
    assert {"balanced_accuracy", "macro_f1", "training_seconds"}.issubset(summary.columns)
    assert summary["training_seconds"].ge(0).all()
    svm_row = summary.loc[summary["model"] == "SVM"].iloc[0]
    assert svm_row["training_samples"] == 60
    for model_name in ("RandomForest", "SVM", "MLP"):
        confusion = pd.read_csv(tmp_path / f"{model_name}_confusion_matrix.csv", index_col=0)
        assert confusion.shape == (6, 6)
        assert (tmp_path / f"{model_name}_confusion_matrix.png").stat().st_size > 1000
        assert (tmp_path / f"{model_name}.joblib").exists()
    svm_model = joblib.load(tmp_path / "SVM.joblib")
    mlp_model = joblib.load(tmp_path / "MLP.joblib")
    assert list(svm_model.named_steps) == ["scaler", "model"]
    assert list(mlp_model.named_steps) == ["scaler", "model"]


def test_full_modeling_pipeline_writes_auditable_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]

    def make_frame(prefix: str, files_per_label: int) -> pd.DataFrame:
        rows = []
        for label_index, label in enumerate(labels):
            for file_index in range(files_per_label):
                for window_index in range(4):
                    row = _metadata_row(prefix, label_index * 10 + file_index, 0, window_index, label)
                    for feature_index, feature in enumerate(FEATURE_COLUMNS):
                        row[feature] = label_index + rng.normal(0, 0.1) if feature_index == 0 else rng.normal()
                    rows.append(row)
        return pd.DataFrame(rows)

    feature_root = tmp_path / "features"
    output_root = tmp_path / "models"
    feature_root.mkdir()
    make_frame("train", 5).to_csv(feature_root / "train_features_raw.csv", index=False)
    make_frame("test", 1).to_csv(feature_root / "test_features_raw.csv", index=False)
    config = ModelingConfig(
        feature_root=feature_root,
        output_root=output_root,
        top_k_candidates=(2, 3),
        rf_n_estimators=10,
        rf_n_jobs=1,
        svm_sample_size=60,
        mlp_max_iter=20,
        mlp_n_iter_no_change=5,
        plot_dpi=80,
    )

    summary = run_full_modeling(config)

    assert summary["leakage_check"]["file_path_overlap_count"] == 0
    assert summary["leakage_check"]["run_id_overlap_count"] == 0
    assert summary["selected_k"] in {2, 3}
    assert len(summary["selected_features"]) == summary["selected_k"]
    assert (output_root / "selected_features.json").exists()
    assert (output_root / "removed_features.csv").exists()
    assert (output_root / "top_k_cv_summary.csv").exists()
    assert (output_root / "feature_importance.csv").exists()
    assert (output_root / "model_metrics.csv").exists()
    assert (output_root / "run_complete.json").exists()


def test_correlation_reassessment_writes_svm_only_comparisons(tmp_path: Path) -> None:
    rng = np.random.default_rng(321)
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
    feature_root = tmp_path / "features"
    output_root = tmp_path / "reassess"
    feature_root.mkdir()

    def make_frame(prefix: str, file_offset: int) -> pd.DataFrame:
        rows = []
        for label_index, label in enumerate(labels):
            for file_index in range(2):
                for window_index in range(3):
                    row = _metadata_row(prefix, file_offset + label_index * 10 + file_index, 0, window_index, label)
                    for feature_index, feature in enumerate(FEATURE_COLUMNS):
                        value = label_index + rng.normal(0, 0.02)
                        if feature == "std":
                            value = label_index + rng.normal(0, 0.02)
                        if feature == "mean":
                            value = label_index + rng.normal(0, 0.02)
                        row[feature] = value
                    rows.append(row)
        frame = pd.DataFrame(rows)
        frame["std"] = frame["mean"] * 1.01 + rng.normal(0, 1e-4, size=len(frame))
        return frame

    make_frame("train", 0).to_csv(feature_root / "train_features_raw.csv", index=False)
    make_frame("test", 100).to_csv(feature_root / "test_features_raw.csv", index=False)
    pd.DataFrame(
        {
            "file_path": ["/train/a.csv", "/train/b.csv"],
            "condition_id": ["cond_a", "cond_b"],
            "sha256": ["hash_a", "hash_b"],
        }
    ).to_csv(feature_root / "train_manifest.csv", index=False)
    pd.DataFrame(
        {
            "file_path": ["/test/a.csv", "/test/b.csv"],
            "condition_id": ["cond_c", "cond_d"],
            "sha256": ["hash_c", "hash_d"],
        }
    ).to_csv(feature_root / "test_manifest.csv", index=False)

    config = ModelingConfig(
        feature_root=feature_root,
        output_root=output_root,
        rf_n_estimators=10,
        rf_n_jobs=1,
        svm_sample_size=24,
        mlp_max_iter=15,
        mlp_n_iter_no_change=5,
        plot_dpi=80,
    )

    summary = run_correlation_reassessment(config)

    assert summary["leakage_check"]["file_path_overlap_count"] == 0
    assert (output_root / "data_leakage_check.csv").exists()
    assert (output_root / "all_valid_features.csv").exists()
    assert (output_root / "corr_095_before.csv").exists()
    assert (output_root / "corr_095_after.csv").exists()
    assert (output_root / "corr_090_before.csv").exists()
    assert (output_root / "corr_090_after.csv").exists()
    comparison = pd.read_csv(output_root / "model_comparison.csv")
    assert set(comparison["feature_set"]) == {"all_valid", "corr_095", "corr_090"}
    assert set(comparison["model"]) == {"SVM"}
    assert {"accuracy", "balanced_accuracy", "macro_f1"}.issubset(comparison.columns)
    assert {"recall_松动", "recall_轴承故障", "recall_联轴器不对中"}.issubset(comparison.columns)
