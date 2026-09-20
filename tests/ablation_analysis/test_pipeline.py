import json

import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_40, FEATURE_43
from ablation_analysis.iteration_selection import make_record_folds
from ablation_analysis.modeling import train_selected_model
from ablation_analysis.pipeline import _validate_baseline_predictions, build_run_matrix, run_pipeline


def test_build_run_matrix_has_exact_deterministic_twelve_runs():
    matrix = build_run_matrix()
    assert matrix.to_dict("records") == [
        {"channel": channel, "split_mode": mode, "feature_set": feature_set}
        for channel in (3, 4, 5)
        for mode in ("record", "temporal")
        for feature_set in ("features_43", "features_40")
    ]


def _tiny_frame():
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
    rows = []
    for class_index, label in enumerate(labels):
        for record_index in range(5):
            for window_index in range(2):
                rows.append(
                    {
                        "record_id": f"r{class_index}-{record_index}",
                        "window_id": window_index,
                        "label": label,
                        "motor": "M",
                        "rpm": 1000,
                        "condition": "c",
                        "state": "s",
                        "severity": "none",
                        "start_sample": window_index * 10,
                        "end_sample": window_index * 10 + 10,
                        "split": "train_dev",
                        **{name: class_index + record_index / 10 + window_index / 100 + i / 1000 for i, name in enumerate(FEATURE_43)},
                    }
                )
    frame = pd.DataFrame(rows)
    return frame


def test_train_selected_model_enforces_iteration_and_feature_contract(tmp_path):
    frame = _tiny_frame()
    frame.loc[frame.groupby("label").cumcount().ge(8), "split"] = "test"
    train = frame[frame.split.eq("train_dev")].reset_index(drop=True)
    folds = make_record_folds(train.label, train.record_id, n_splits=2)
    cv = pd.DataFrame({"fold": [1, 2], "iteration": [2, 2], "record_macro_f1": [0.5, 0.6]})
    summary = pd.DataFrame({"iteration": [2], "mean_record_macro_f1": [0.55], "std_record_macro_f1": [0.1], "fold_count": [2]})
    with pytest.raises(ValueError, match="ITERATION_GRID"):
        train_selected_model(frame, "record", 3, FEATURE_40, 3, folds[2], cv, summary, tmp_path / "bad")
    with pytest.raises(ValueError, match="exactly FEATURE_43 or FEATURE_40"):
        train_selected_model(frame, "record", 3, FEATURE_40[:-1], 20, folds[2], cv, summary, tmp_path / "bad2")


def test_pipeline_refuses_existing_output_before_validation(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        run_pipeline(tmp_path / "missing-features", tmp_path / "missing-baseline", output)
    assert marker.read_text() == "untouched"
    assert list(output.iterdir()) == [marker]


def test_baseline_prediction_contract_rejects_missing_or_unknown_predictions():
    frame = pd.DataFrame({
        "label": ["正常"],
        **{label: [1.0 if label == "正常" else 0.0] for label in ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]},
    })
    with pytest.raises(ValueError, match="predicted_label"):
        _validate_baseline_predictions(frame)
    frame["predicted_label"] = "unknown"
    with pytest.raises(ValueError, match="unknown predicted_label"):
        _validate_baseline_predictions(frame)


def test_tiny_end_to_end_runs_all_paired_models_without_touching_sources(tmp_path, monkeypatch):
    source = tmp_path / "features"
    baseline = tmp_path / "baseline"
    tables = {"record": {}, "temporal": {}}
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
    for mode, folder in (("record", "file_split"), ("temporal", "temporal_split")):
        (source / folder).mkdir(parents=True, exist_ok=True)
        for channel in (3, 4, 5):
            rows = []
            for class_index, label in enumerate(labels):
                for record_index in range(6):
                    split = ("train_dev" if mode == "record" else "train") if record_index < 5 else "test"
                    rows.append(
                        {
                            "record_id": f"{mode}-{class_index}-{record_index}",
                            "window_id": 0,
                            "label": label,
                            "motor": "M",
                            "rpm": 1000,
                            "condition": "c",
                            "state": "s",
                            "severity": "none",
                            "start_sample": record_index * 10,
                            "end_sample": record_index * 10 + 10,
                            "split": split,
                            "channel": channel,
                            **{
                                name: class_index * 10 + record_index + channel / 10 + feature_index / 1000
                                for feature_index, name in enumerate(FEATURE_43)
                            },
                        }
                    )
            data = pd.DataFrame(rows)
            tables[mode][channel] = data
            data.to_csv(source / folder / f"features_ch{channel}.csv", index=False)
            prediction_dir = baseline / "models" / mode / f"ch{channel}"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            predictions = data.loc[data["split"].eq("test"), [
                    "record_id", "window_id", "label", "motor", "rpm", "condition", "state",
                    "severity", "start_sample", "end_sample", "split", "channel",
            ]].copy()
            predictions["predicted_label"] = predictions["label"]
            for label in labels:
                predictions[label] = predictions["label"].eq(label).astype(float)
            predictions.to_csv(prediction_dir / "window_predictions.csv", index=False)

    def accepted(_source):
        return pd.DataFrame([{"check": "injected", "measured": 0, "threshold": 0, "passed": True, "detail": ""}]), tables

    pairs = []
    for index in range(19):
        pairs.append({
            "feature_a": FEATURE_43[index], "feature_b": FEATURE_43[index + 1],
            "pearson_r": 0.96, "channel": 3 + index % 3,
            "split_mode": "record" if index % 2 == 0 else "temporal",
        })
    correlation_dir = baseline / "correlations"
    correlation_dir.mkdir(parents=True)
    pd.DataFrame(pairs).to_csv(correlation_dir / "pearson_high_correlation_pairs.csv", index=False)
    before = {
        path: path.read_bytes()
        for path in [*source.rglob("*.csv"), *baseline.rglob("*.csv")]
    }

    import ablation_analysis.config as ablation_config
    import ablation_analysis.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "accept_feature_tables", accepted)
    monkeypatch.setattr(pipeline_module, "ITERATION_GRID", [2, 4])
    monkeypatch.setattr(pipeline_module, "MAX_ITERATIONS", 4)
    monkeypatch.setattr(ablation_config, "ITERATION_GRID", [2, 4])

    output = tmp_path / "output"
    result = run_pipeline(source, baseline, output)
    assert len(result) == 12
    run_dirs = list((output / "models").glob("*/ch*/features_*"))
    assert len(run_dirs) == 12
    required = {
        "model.cbm", "metadata.json", "internal_cv_fold_scores.csv",
        "internal_cv_iteration_summary.csv", "window_predictions.csv",
        "record_predictions.csv", "window_metrics.json", "record_metrics.json",
        "feature_importance.csv",
    }
    for run_dir in run_dirs:
        assert required.issubset({path.name for path in run_dir.iterdir()})
        metadata = json.loads((run_dir / "metadata.json").read_text())
        assert metadata["selected_iteration"] in [2, 4]
        assert metadata["train_windows"] == 30
        assert metadata["test_windows"] == 6
        assert metadata["train_records"] == 30
        assert metadata["test_records"] == 6
        assert metadata["confidence"]["window_prediction_count"] == 6
        assert metadata["confidence"]["record_prediction_count"] == 6
        records = pd.read_csv(run_dir / "record_predictions.csv")
        assert {"window_count", "valid_window_count", "mean_max_class_probability"}.issubset(records.columns)
        assert records["valid_window_count"].eq(records["window_count"]).all()
        assert records["mean_max_class_probability"].between(0, 1).all()
        assert (run_dir / "model.cbm").stat().st_size > 0
    for channel in (3, 4, 5):
        for mode in ("record", "temporal"):
            full = json.loads((output / "models" / mode / f"ch{channel}" / "features_43" / "metadata.json").read_text())
            reduced = json.loads((output / "models" / mode / f"ch{channel}" / "features_40" / "metadata.json").read_text())
            assert full["fold_sha256"] == reduced["fold_sha256"]
            manifest = pd.read_csv(output / "fold_manifests" / f"{mode}_ch{channel}.csv")
            training = tables[mode][channel].query("split == @full['training_split']")
            assert set(manifest.record_id) == set(training.record_id)
            assert set(manifest.role) == {"fit", "validation"}
            test_records = set(tables[mode][channel].query("split == 'test'").record_id)
            assert not set(manifest.record_id) & test_records
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["iteration_grid"] == [2, 4]
    assert run_manifest["max_iterations"] == 4
    assert run_manifest["cv_splits"] == 5
    assert run_manifest["input_sha256"]
    assert all(run_manifest["input_sha256"].values())
    report_files = {
        "diagnostics/four_group_feature_statistics.csv",
        "diagnostics/record_level_cliffs_delta.csv",
        "diagnostics/targeted_error_records.csv",
        "diagnostics/error_concentration.csv",
        "redundancy/consolidated_high_correlation_pairs.csv",
        "redundancy/feature_decisions_43_to_40.csv",
        "iteration_selection/all_iteration_curves.csv",
        "comparison/ablation_metrics.csv",
        "comparison/ablation_deltas.csv",
        "comparison/channel_class_recall.csv",
        "comparison/class_recall_deltas.csv",
        "comparison/ch5_unique_value.csv",
        "comparison/new_feature_decision.csv",
        "comparison/new_feature_evidence.csv",
        "figures/font_metadata.json",
        "conclusion.md",
    }
    assert all((output / relative).is_file() for relative in report_files)
    assert len(list((output / "figures" / "iteration_curves").glob("*.png"))) == 12
    assert len(list((output / "figures" / "class_recall").glob("*.png"))) >= 4
    assert {path: path.read_bytes() for path in before} == before

    original_train = pipeline_module.train_selected_model

    def fail_late(*args, **kwargs):
        raise RuntimeError("simulated late training failure")

    monkeypatch.setattr(pipeline_module, "train_selected_model", fail_late)
    failed_output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="staging retained"):
        run_pipeline(source, baseline, failed_output)
    assert not failed_output.exists()
    assert len(list(tmp_path.glob("failed-output.staging-*"))) == 1
    with pytest.raises(RuntimeError, match="staging retained"):
        run_pipeline(source, baseline, failed_output)
    assert not failed_output.exists()
    assert len(list(tmp_path.glob("failed-output.staging-*"))) == 2

    monkeypatch.setattr(pipeline_module, "train_selected_model", original_train)
    original_hash_inputs = pipeline_module._hash_inputs
    hash_calls = 0

    def changed_hash_snapshot(paths):
        nonlocal hash_calls
        hash_calls += 1
        hashes = original_hash_inputs(paths)
        if hash_calls == 3:
            first = next(iter(hashes))
            hashes[first] = "0" * 64
        return hashes

    monkeypatch.setattr(pipeline_module, "_hash_inputs", changed_hash_snapshot)
    changed_output = tmp_path / "changed-output"
    with pytest.raises(RuntimeError, match="input hash mismatch"):
        run_pipeline(source, baseline, changed_output)
    assert not changed_output.exists()
    assert len(list(tmp_path.glob("changed-output.staging-*"))) == 1

    monkeypatch.setattr(pipeline_module, "_hash_inputs", original_hash_inputs)
    original_read_csv = pd.read_csv

    def missing_prediction_column(path, *args, **kwargs):
        frame = original_read_csv(path, *args, **kwargs)
        if str(path).endswith("window_predictions.csv"):
            return frame.drop(columns="predicted_label")
        return frame

    monkeypatch.setattr(pipeline_module.pd, "read_csv", missing_prediction_column)
    invalid_output = tmp_path / "invalid-output"
    with pytest.raises(ValueError, match="predicted_label"):
        run_pipeline(source, baseline, invalid_output)
    assert not invalid_output.exists()
    assert not list(tmp_path.glob("invalid-output.staging-*"))
