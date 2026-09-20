import json
from pathlib import Path

import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_43
from ablation_analysis.iteration_selection import _manifest_hash
from ablation_analysis.pipeline import run_pipeline
from ablation_analysis.verification import _validate_temporal_rows, verify_output


@pytest.mark.parametrize("kind,match", [("guard", "guard"), ("block", "block boundary"), ("overlap", "declared range")])
def test_temporal_protocol_rejects_guard_block_and_overlapping_nonidentical_windows(kind, match):
    bounds = pd.DataFrame([{"record_id": "r", "train_start": 20000, "train_end": 58400,
                            "guard_start": 8000, "guard_end": 20000, "test_start": 0, "test_end": 8000}]).set_index("record_id")
    rows = pd.DataFrame([{"record_id": "r", "split": "train", "start_sample": 20000, "end_sample": 24800},
                         {"record_id": "r", "split": "test", "start_sample": 0, "end_sample": 4800}])
    if kind == "guard": rows.loc[1, ["start_sample", "end_sample"]] = [7000, 11800]
    elif kind == "block": rows.loc[0, ["start_sample", "end_sample"]] = [37000, 41800]
    else: rows.loc[1, ["start_sample", "end_sample"]] = [21000, 23000]
    with pytest.raises(ValueError, match=match):
        _validate_temporal_rows(rows, bounds)


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
                    start_sample = record_index * 10 if mode == "record" else (20000 + record_index * 10 if split == "train" else 0)
                    rows.append({
                        "record_id": record_id, "window_id": record_index,
                        "label": label, "motor": "M", "rpm": 1000, "condition": "c",
                        "state": "s", "severity": "none", "start_sample": start_sample,
                        "end_sample": start_sample + 10, "split": split, "channel": channel,
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
    temporal_records = sorted(tables["temporal"][3].record_id.unique())
    pd.DataFrame([{"record_id": record_id, "train_start": 20000, "train_end": 39200,
                   "guard_start": 8000, "guard_end": 20000, "test_start": 0, "test_end": 8000,
                   "split_direction": "test_head"} for record_id in temporal_records]).to_csv(
        source / "temporal_split" / "temporal_split.csv", index=False)
    correlation_dir = baseline / "correlations"
    correlation_dir.mkdir(parents=True)
    pd.DataFrame([{
        "feature_a": FEATURE_43[i], "feature_b": FEATURE_43[i + 1], "pearson_r": .96,
        "channel": 3 + i % 3, "split_mode": "record" if i % 2 == 0 else "temporal",
    } for i in range(19)]).to_csv(correlation_dir / "pearson_high_correlation_pairs.csv", index=False)

    import ablation_analysis.config as config
    import ablation_analysis.pipeline as pipeline
    import ablation_analysis.verification as verification
    accepted = lambda _: (
        pd.DataFrame([{"check": "fixture", "measured": 0, "threshold": 0, "passed": True, "detail": ""}]), tables
    )
    monkeypatch.setattr(pipeline, "accept_feature_tables", accepted)
    monkeypatch.setattr(verification, "accept_feature_tables", accepted)
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
    ("wrong_valid_iteration", "chosen iteration"),
    ("fold_overlap", "fold"),
    ("test_intrusion", "fold"),
    ("fold_hash", "fold"),
    ("invalid_probability", "probability"),
    ("invalid_record_probability", "record predictions"),
    ("changed_comparison", "comparison"),
    ("missing_report", "report"),
    ("missing_diagnostic", "report"),
    ("changed_diagnostic", "diagnostic"),
    ("coherent_label", "authoritative label"),
    ("replacement_png", "pixel mismatch"),
])
def test_verifier_rejects_corrupted_outputs(tmp_path, monkeypatch, mutation, match):
    output = _build_fixture(tmp_path, monkeypatch)
    run = output / "models" / "record" / "ch3" / "features_40"
    if mutation == "missing_model":
        (run / "model.cbm").rename(run / "model.cbm.hidden")
    elif mutation == "non_grid_iteration":
        path = run / "metadata.json"; value = json.loads(path.read_text()); value["selected_iteration"] = 810
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    elif mutation == "wrong_valid_iteration":
        path = run / "metadata.json"; value = json.loads(path.read_text()); value["selected_iteration"] = 2 if value["selected_iteration"] == 4 else 4
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        comparison = output / "comparison" / "ablation_metrics.csv"; frame = pd.read_csv(comparison)
        mask = frame.split_mode.eq("record") & frame.channel.eq(3) & frame.feature_set.eq("features_40")
        frame.loc[mask, "selected_iteration"] = value["selected_iteration"]; frame.to_csv(comparison, index=False)
    elif mutation == "fold_overlap":
        path = output / "fold_manifests" / "record_ch3.csv"; frame = pd.read_csv(path)
        frame.loc[(frame.fold == 1) & (frame.role == "validation"), "role"] = "fit"
        frame["fold_sha256"] = _manifest_hash(frame)
        frame.to_csv(path, index=False)
    elif mutation == "test_intrusion":
        path = output / "fold_manifests" / "record_ch3.csv"; frame = pd.read_csv(path)
        source = Path(json.loads((output / "run_manifest.json").read_text())["feature_source_root"])
        test_record = pd.read_csv(source / "file_split" / "features_ch3.csv").query("split == 'test'").record_id.iloc[0]
        frame.loc[0, "record_id"] = test_record; frame["fold_sha256"] = _manifest_hash(frame)
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
    elif mutation == "invalid_record_probability":
        path = run / "record_predictions.csv"; frame = pd.read_csv(path)
        frame.loc[0, "正常"] += .2; frame.to_csv(path, index=False)
    elif mutation == "missing_report":
        path = output / "conclusion.md"; path.rename(output / "conclusion.md.hidden")
    elif mutation in {"missing_diagnostic", "changed_diagnostic"}:
        path = output / "diagnostics" / "error_concentration.csv"
        if mutation == "missing_diagnostic":
            path.rename(path.with_suffix(".csv.hidden"))
        else:
            frame = pd.read_csv(path); frame.loc[0, "top5_share"] += .1; frame.to_csv(path, index=False)
    elif mutation == "coherent_label":
        path = run / "window_predictions.csv"; frame = pd.read_csv(path); frame.loc[0, "label"] = "汽蚀"; frame.to_csv(path, index=False)
        path = run / "record_predictions.csv"; frame = pd.read_csv(path); frame.loc[0, "label"] = "汽蚀"; frame.to_csv(path, index=False)
    else:
        from PIL import Image as PILImage
        path = output / "figures" / "iteration_curves" / "ch3_record_features_40.png"
        with PILImage.open(path) as image:
            replacement = PILImage.new("RGB", image.size, "white")
        replacement.save(path)
    with pytest.raises(ValueError, match=match):
        verify_output(output, deep=False)
