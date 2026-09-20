import json

import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_40, FEATURE_43
from ablation_analysis.iteration_selection import make_record_folds
from ablation_analysis.modeling import train_selected_model
from ablation_analysis.pipeline import build_run_matrix, run_pipeline


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
    with pytest.raises(ValueError, match="iteration grid"):
        train_selected_model(frame, "record", 3, FEATURE_40, 3, folds[2], cv, summary, tmp_path / "bad", iteration_grid=[2, 4])
    with pytest.raises(ValueError, match="exactly FEATURE_43 or FEATURE_40"):
        train_selected_model(frame, "record", 3, FEATURE_40[:-1], 2, folds[2], cv, summary, tmp_path / "bad2", iteration_grid=[2, 4])


def test_pipeline_refuses_existing_output_before_validation(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        run_pipeline(tmp_path / "missing-features", tmp_path / "missing-baseline", output)
    assert marker.read_text() == "untouched"
    assert list(output.iterdir()) == [marker]


def test_tiny_end_to_end_runs_all_paired_models_without_touching_sources(tmp_path):
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
            predictions = data[
                [
                    "record_id", "window_id", "label", "motor", "rpm", "condition", "state",
                    "severity", "start_sample", "end_sample", "split", "channel",
                ]
            ].copy()
            predictions["predicted_label"] = predictions["label"]
            predictions.to_csv(prediction_dir / "window_predictions.csv", index=False)

    before = {
        path: path.read_bytes()
        for path in [*source.rglob("*.csv"), *baseline.rglob("*.csv")]
    }

    def accepted(_source):
        return pd.DataFrame([{"check": "injected", "measured": 0, "threshold": 0, "passed": True, "detail": ""}]), tables

    output = tmp_path / "output"
    result = run_pipeline(
        source,
        baseline,
        output,
        checkpoints=[2, 4],
        max_iterations=4,
        acceptance_loader=accepted,
    )
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
        assert (run_dir / "model.cbm").stat().st_size > 0
    for channel in (3, 4, 5):
        for mode in ("record", "temporal"):
            full = json.loads((output / "models" / mode / f"ch{channel}" / "features_43" / "metadata.json").read_text())
            reduced = json.loads((output / "models" / mode / f"ch{channel}" / "features_40" / "metadata.json").read_text())
            assert full["fold_sha256"] == reduced["fold_sha256"]
            manifest = pd.read_csv(output / "fold_manifests" / f"{mode}_ch{channel}.csv")
            training = tables[mode][channel].query("split == @full['training_split']")
            assert set(manifest.record_id) == set(training.record_id)
    assert {path: path.read_bytes() for path in before} == before
