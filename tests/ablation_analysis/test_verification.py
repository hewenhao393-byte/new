import json
from pathlib import Path

import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_43
from ablation_analysis.pipeline import run_pipeline
from ablation_analysis.verification import verify_output


def _build_fixture(tmp_path, monkeypatch):
    source, baseline = tmp_path / "features", tmp_path / "baseline"
    tables = {"record": {}, "temporal": {}}
    labels = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
    for mode, folder in (("record", "file_split"), ("temporal", "temporal_split")):
        (source / folder).mkdir(parents=True)
        for channel in (3, 4, 5):
            rows = []
            for class_index, label in enumerate(labels):
                for record_index in range(6):
                    split = ("train_dev" if mode == "record" else "train") if record_index < 5 else "test"
                    record_id = f"{mode}-{class_index}-{record_index}"
                    if mode == "temporal" and split == "test":
                        record_id = f"{mode}-{class_index}-0"
                    rows.append({
                        "record_id": record_id, "window_id": record_index,
                        "label": label, "motor": "M", "rpm": 1000, "condition": "c",
                        "state": "s", "severity": "none", "start_sample": record_index * 10,
                        "end_sample": record_index * 10 + 10, "split": split, "channel": channel,
                        **{name: class_index * 10 + record_index + channel / 10 + i / 1000
                           for i, name in enumerate(FEATURE_43)},
                    })
            data = pd.DataFrame(rows)
            tables[mode][channel] = data
            data.to_csv(source / folder / f"features_ch{channel}.csv", index=False)
            prediction_dir = baseline / "models" / mode / f"ch{channel}"
            prediction_dir.mkdir(parents=True)
            predictions = data[data.split.eq("test")][[
                "record_id", "window_id", "label", "motor", "rpm", "condition", "state",
                "severity", "start_sample", "end_sample", "split", "channel",
            ]].copy()
            predictions["predicted_label"] = predictions["label"]
            for label_name in labels:
                predictions[label_name] = predictions["label"].eq(label_name).astype(float)
            predictions.to_csv(prediction_dir / "window_predictions.csv", index=False)
    correlation_dir = baseline / "correlations"
    correlation_dir.mkdir(parents=True)
    pd.DataFrame([{
        "feature_a": FEATURE_43[i], "feature_b": FEATURE_43[i + 1], "pearson_r": .96,
        "channel": 3 + i % 3, "split_mode": "record" if i % 2 == 0 else "temporal",
    } for i in range(19)]).to_csv(correlation_dir / "pearson_high_correlation_pairs.csv", index=False)

    import ablation_analysis.config as config
    import ablation_analysis.pipeline as pipeline
    monkeypatch.setattr(pipeline, "accept_feature_tables", lambda _: (
        pd.DataFrame([{"check": "fixture", "measured": 0, "threshold": 0, "passed": True, "detail": ""}]), tables
    ))
    monkeypatch.setattr(pipeline, "ITERATION_GRID", [2, 4])
    monkeypatch.setattr(pipeline, "MAX_ITERATIONS", 4)
    monkeypatch.setattr(config, "ITERATION_GRID", [2, 4])
    output = tmp_path / "output"
    run_pipeline(source, baseline, output)
    return output


def test_verifier_accepts_complete_output_and_writes_detailed_reports(tmp_path, monkeypatch):
    output = _build_fixture(tmp_path, monkeypatch)
    checks = verify_output(output)
    assert checks["passed"].all()
    assert len(checks) >= 100
    assert (output / "verification_report.csv").is_file()
    assert "PASS" in (output / "verification_summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation,match", [
    ("missing_model", "model"),
    ("non_grid_iteration", "iteration"),
    ("fold_overlap", "fold"),
    ("fold_hash", "fold"),
    ("invalid_probability", "probability"),
    ("changed_comparison", "comparison"),
    ("missing_report", "report"),
    ("missing_diagnostic", "report"),
])
def test_verifier_rejects_corrupted_outputs(tmp_path, monkeypatch, mutation, match):
    output = _build_fixture(tmp_path, monkeypatch)
    run = output / "models" / "record" / "ch3" / "features_40"
    if mutation == "missing_model":
        (run / "model.cbm").rename(run / "model.cbm.hidden")
    elif mutation == "non_grid_iteration":
        path = run / "metadata.json"; value = json.loads(path.read_text()); value["selected_iteration"] = 3
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    elif mutation == "fold_overlap":
        path = output / "fold_manifests" / "record_ch3.csv"; frame = pd.read_csv(path)
        frame.loc[(frame.fold == 1) & (frame.role == "validation"), "role"] = "fit"
        frame.to_csv(path, index=False)
    elif mutation == "fold_hash":
        path = output / "fold_manifests" / "record_ch3.csv"; frame = pd.read_csv(path)
        frame["fold_sha256"] = "0" * 64; frame.to_csv(path, index=False)
    elif mutation == "invalid_probability":
        path = run / "window_predictions.csv"; frame = pd.read_csv(path)
        frame.loc[0, ["正常", "转子不平衡"]] = [0.8, 0.8]
        frame.to_csv(path, index=False)
    elif mutation == "changed_comparison":
        path = output / "comparison" / "ablation_metrics.csv"; frame = pd.read_csv(path)
        frame.loc[0, "test_record_macro_f1"] += .01; frame.to_csv(path, index=False)
    elif mutation == "missing_report":
        path = output / "conclusion.md"; path.rename(output / "conclusion.md.hidden")
    else:
        path = output / "diagnostics" / "error_concentration.csv"
        path.rename(path.with_suffix(".csv.hidden"))
    with pytest.raises(ValueError, match=match):
        verify_output(output)
