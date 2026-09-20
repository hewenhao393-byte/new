"""Independent, source-backed verification of a completed ablation experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from PIL import Image

from baseline_analysis.evaluation import evaluate_predictions, fuse_records

from .config import FEATURE_40, FEATURE_43, JOIN_KEYS, LABEL_ORDER
from .iteration_selection import _validate_manifest, aggregate_cv_scores
from .reporting import REPORT_REQUIRED_FILES, _build_class_recall_deltas, _build_deltas


MODEL_FILES = {
    "model.cbm", "metadata.json", "internal_cv_fold_scores.csv",
    "internal_cv_iteration_summary.csv", "window_predictions.csv",
    "record_predictions.csv", "window_metrics.json", "record_metrics.json",
    "feature_importance.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _same_json_numbers(actual, expected, name: str) -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"metric mismatch in {name}: keys differ")
        for key in expected:
            _same_json_numbers(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"metric mismatch in {name}: lengths differ")
        for index, value in enumerate(expected):
            _same_json_numbers(actual[index], value, f"{name}[{index}]")
    elif isinstance(expected, (int, float)) and expected is not None:
        if not np.isclose(float(actual), float(expected), atol=1e-12, rtol=0):
            raise ValueError(f"metric mismatch in {name}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"metric mismatch in {name}: {actual!r} != {expected!r}")


def _assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, keys, name: str) -> None:
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"comparison mismatch in {name}: shape or columns differ")
    left = actual.sort_values(keys, kind="stable").reset_index(drop=True)[expected.columns]
    right = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(left[column].to_numpy(float), right[column].to_numpy(float), atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f"comparison mismatch in {name}.{column}")
        elif not left[column].fillna("<NA>").equals(right[column].fillna("<NA>")):
            raise ValueError(f"comparison mismatch in {name}.{column}")


def verify_output(output_root) -> pd.DataFrame:
    """Verify every persisted result against inputs and independently write an audit report."""
    root = Path(output_root)
    checks = []

    def passed(category, item, detail=""):
        checks.append({"category": category, "item": item, "passed": True, "detail": detail})

    if not root.is_dir():
        raise ValueError(f"report output directory missing: {root}")
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("report run_manifest.json missing")
    manifest = _json(manifest_path)
    source = Path(manifest["feature_source_root"])
    baseline = Path(manifest["baseline_root"])
    grid = [int(value) for value in manifest["iteration_grid"]]
    if not grid or grid != sorted(set(grid)) or any(value < 1 for value in grid):
        raise ValueError("iteration grid is not canonical")
    passed("manifest", "iteration_grid", json.dumps(grid))

    for path_string, expected_hash in manifest["input_sha256"].items():
        path = Path(path_string)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"input hash mismatch: {path}")
        passed("input_hash", str(path), expected_hash)

    run_dirs = sorted((root / "models").glob("*/ch*/features_*"))
    model_files = sorted((root / "models").glob("*/ch*/features_*/model.cbm"))
    if len(run_dirs) != 12 or len(model_files) != 12:
        raise ValueError(f"model coverage mismatch: run_dirs={len(run_dirs)}, models={len(model_files)}")
    expected_runs = {(mode, channel, fs) for mode in ("record", "temporal") for channel in (3, 4, 5)
                     for fs in ("features_43", "features_40")}
    actual_runs = {(path.parts[-3], int(path.parts[-2][2:]), path.name) for path in run_dirs}
    if actual_runs != expected_runs:
        raise ValueError("model run directory coverage mismatch")
    passed("coverage", "model_runs", "12 exact runs")

    source_tables = {}
    fold_hashes = {}
    for mode, folder, training_split in (("record", "file_split", "train_dev"), ("temporal", "temporal_split", "train")):
        for channel in (3, 4, 5):
            data = pd.read_csv(source / folder / f"features_ch{channel}.csv", low_memory=False)
            source_tables[(mode, channel)] = data
            train = data[data.split.eq(training_split)].reset_index(drop=True)
            test = data[data.split.eq("test")].reset_index(drop=True)
            fold_path = root / "fold_manifests" / f"{mode}_ch{channel}.csv"
            index_path = root / "fold_manifests" / f"{mode}_ch{channel}_indices.npz"
            fold = pd.read_csv(fold_path)
            try:
                fold_hash = _validate_manifest(fold)
            except ValueError as exc:
                raise ValueError(f"fold manifest invalid for {mode} ch{channel}: {exc}") from exc
            if set(fold.record_id) != set(train.record_id):
                raise ValueError(f"fold records do not exactly cover training records for {mode} ch{channel}")
            if mode == "record" and set(fold.record_id) & set(test.record_id):
                raise ValueError(f"test record entered record-mode CV for {mode} ch{channel}")
            train_windows = set(map(tuple, train[JOIN_KEYS].itertuples(index=False, name=None)))
            test_windows = set(map(tuple, test[JOIN_KEYS].itertuples(index=False, name=None)))
            if train_windows & test_windows:
                raise ValueError(f"test window entered CV source rows for {mode} ch{channel}")
            bundle = np.load(index_path)
            folds = sorted(fold.fold.unique())
            if set(bundle.files) != {f"fold_{number}_{role}" for number in folds for role in ("fit", "validation")}:
                raise ValueError(f"fold index keys invalid for {mode} ch{channel}")
            validation_counts = np.zeros(len(train), dtype=int)
            for number in folds:
                fit = np.asarray(bundle[f"fold_{number}_fit"])
                valid = np.asarray(bundle[f"fold_{number}_validation"])
                if set(fit) & set(valid) or set(fit) | set(valid) != set(range(len(train))):
                    raise ValueError(f"fold overlap or incomplete indices for {mode} ch{channel} fold {number}")
                if set(train.iloc[fit].record_id) != set(fold.query("fold == @number and role == 'fit'").record_id):
                    raise ValueError(f"fold fit indices mismatch for {mode} ch{channel}")
                if set(train.iloc[valid].record_id) != set(fold.query("fold == @number and role == 'validation'").record_id):
                    raise ValueError(f"fold validation indices mismatch for {mode} ch{channel}")
                validation_counts[valid] += 1
            if not np.all(validation_counts == 1):
                raise ValueError(f"fold validation coverage mismatch for {mode} ch{channel}")
            fold_hashes[(mode, channel)] = fold_hash
            passed("fold", f"{mode}_ch{channel}", fold_hash)

            baseline_predictions = pd.read_csv(
                baseline / "models" / mode / f"ch{channel}" / "window_predictions.csv", low_memory=False
            )
            source_keys = set(map(tuple, test[JOIN_KEYS].itertuples(index=False, name=None)))
            prediction_keys = set(map(tuple, baseline_predictions[JOIN_KEYS].itertuples(index=False, name=None)))
            if source_keys != prediction_keys or len(test) != len(baseline_predictions):
                raise ValueError(f"diagnostic join unmatched rows for {mode} ch{channel}")
            passed("diagnostic_join", f"{mode}_ch{channel}", "unmatched=0")

    metric_rows, recall_rows = [], []
    for mode, channel, feature_set in sorted(expected_runs):
        run = root / "models" / mode / f"ch{channel}" / feature_set
        missing = MODEL_FILES - {path.name for path in run.iterdir()}
        if missing:
            raise ValueError(f"model files missing in {run}: {sorted(missing)}")
        metadata = _json(run / "metadata.json")
        selected = int(metadata["selected_iteration"])
        if selected not in grid:
            raise ValueError(f"iteration {selected} is not in production grid for {run}")
        if metadata["fold_sha256"] != fold_hashes[(mode, channel)]:
            raise ValueError(f"fold hash mismatch in model metadata for {run}")
        features = FEATURE_43 if feature_set == "features_43" else FEATURE_40
        if metadata["features"] != features:
            raise ValueError(f"feature contract mismatch for {run}")
        summary = pd.read_csv(run / "internal_cv_iteration_summary.csv")
        scores = pd.read_csv(run / "internal_cv_fold_scores.csv")
        if set(summary.iteration.astype(int)) != set(grid) or set(scores.iteration.astype(int)) != set(grid):
            raise ValueError(f"iteration curve does not cover production grid for {run}")
        recomputed_summary = aggregate_cv_scores(scores)
        _assert_frame_close(summary, recomputed_summary, ["iteration"], f"iteration summary {run}")

        data = source_tables[(mode, channel)]
        test = data[data.split.eq("test")].reset_index(drop=True)
        saved = pd.read_csv(run / "window_predictions.csv", low_memory=False)
        probabilities = saved[LABEL_ORDER].to_numpy(float)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError(f"probability values invalid for {run}")
        if list(saved.predicted_label) != [LABEL_ORDER[index] for index in probabilities.argmax(axis=1)]:
            raise ValueError(f"probability argmax mismatch for {run}")
        if not np.allclose(saved["max_class_probability"].to_numpy(float), probabilities.max(axis=1), atol=1e-12, rtol=0):
            raise ValueError(f"mean max confidence source mismatch for {run}")
        if set(map(tuple, saved[JOIN_KEYS].itertuples(index=False, name=None))) != set(
                map(tuple, test[JOIN_KEYS].itertuples(index=False, name=None))):
            raise ValueError(f"prediction coverage mismatch for {run}")
        ordered = saved[JOIN_KEYS].merge(test, on=JOIN_KEYS, how="left", validate="one_to_one")
        model = CatBoostClassifier()
        model.load_model(str(run / "model.cbm"))
        reproduced = np.asarray(model.predict_proba(ordered[features]), float)
        model_classes = list(model.classes_)
        reproduced = reproduced[:, [model_classes.index(label) for label in LABEL_ORDER]]
        if not np.allclose(reproduced, probabilities, atol=1e-12, rtol=0):
            raise ValueError(f"model reload probability mismatch for {run}")
        passed("model", f"{mode}_ch{channel}_{feature_set}", f"iteration={selected}; atol=1e-12")

        records = pd.read_csv(run / "record_predictions.csv", low_memory=False)
        recomputed_records = fuse_records(saved, LABEL_ORDER)
        _assert_frame_close(records, recomputed_records, ["record_id"], f"record predictions {run}")
        if not records.valid_window_count.eq(records.window_count).all():
            raise ValueError(f"valid_window_count mismatch for {run}")
        if not records.mean_max_class_probability.between(0, 1).all():
            raise ValueError(f"mean max confidence invalid for {run}")
        confidence = metadata["confidence"]
        if not np.isclose(confidence["window_mean_max_class_probability"], saved.max_class_probability.mean(), atol=1e-12, rtol=0):
            raise ValueError(f"mean max confidence metadata mismatch for {run}")
        if not np.isclose(confidence["record_mean_max_class_probability"], records.mean_max_class_probability.mean(), atol=1e-12, rtol=0):
            raise ValueError(f"record mean max confidence metadata mismatch for {run}")
        window_metrics = evaluate_predictions(saved.label, saved.predicted_label, LABEL_ORDER)
        record_metrics = evaluate_predictions(records.label, records.predicted_label, LABEL_ORDER)
        _same_json_numbers(_json(run / "window_metrics.json"), window_metrics, f"window metric {run}")
        _same_json_numbers(_json(run / "record_metrics.json"), record_metrics, f"record metric {run}")
        selected_cv = summary.loc[summary.iteration.eq(selected), "mean_record_macro_f1"]
        metric_rows.append({
            "channel": channel, "split_mode": mode, "feature_set": feature_set,
            "selected_iteration": selected,
            "internal_cv_selected_record_macro_f1": float(selected_cv.iloc[0]),
            "test_window_accuracy": window_metrics["accuracy"], "test_window_macro_f1": window_metrics["macro_f1"],
            "test_window_weighted_f1": window_metrics["weighted_f1"],
            "test_record_accuracy": record_metrics["accuracy"], "test_record_macro_f1": record_metrics["macro_f1"],
            "test_record_weighted_f1": record_metrics["weighted_f1"],
        })
        for level, values in (("window", window_metrics), ("record", record_metrics)):
            for label in LABEL_ORDER:
                recall_rows.append({"channel": channel, "split_mode": mode, "feature_set": feature_set,
                                    "evaluation_level": level, "class": label,
                                    "recall": values["classification_report"][label]["recall"]})
                passed("metric", f"{mode}_ch{channel}_{feature_set}_{level}_{label}")

    metrics, recalls = pd.DataFrame(metric_rows), pd.DataFrame(recall_rows)
    if len(metrics) != 12 or len(recalls) != 144:
        raise ValueError(f"comparison coverage mismatch: metrics={len(metrics)}, recalls={len(recalls)}")
    _assert_frame_close(pd.read_csv(root / "comparison" / "ablation_metrics.csv"), metrics,
                        ["channel", "split_mode", "feature_set"], "comparison metrics")
    _assert_frame_close(pd.read_csv(root / "comparison" / "channel_class_recall.csv"), recalls,
                        ["channel", "split_mode", "feature_set", "evaluation_level", "class"], "comparison recalls")
    deltas = _build_deltas(metrics, recalls)
    _assert_frame_close(pd.read_csv(root / "comparison" / "ablation_deltas.csv"), deltas,
                        ["channel", "split_mode"], "comparison deltas")
    recall_deltas = _build_class_recall_deltas(recalls)
    persisted_recall_deltas = pd.read_csv(root / "comparison" / "class_recall_deltas.csv")
    if len(persisted_recall_deltas) != 72:
        raise ValueError(f"comparison recall delta coverage must be exactly 72 rows, got {len(persisted_recall_deltas)}")
    _assert_frame_close(persisted_recall_deltas, recall_deltas,
                        ["channel", "split_mode", "evaluation_level", "class"], "comparison recall deltas")
    passed("coverage", "class_recall_deltas", "72 exact rows")

    missing_reports = [relative for relative in REPORT_REQUIRED_FILES if not (root / relative).is_file()]
    if missing_reports:
        raise ValueError(f"report files missing: {missing_reports}")
    passed("report", "required_files", str(len(REPORT_REQUIRED_FILES)))
    iteration_pngs = sorted((root / "figures" / "iteration_curves").glob("*.png"))
    recall_pngs = sorted((root / "figures" / "class_recall").glob("*.png"))
    if len(iteration_pngs) != 12 or len(recall_pngs) != 4:
        raise ValueError(f"report figure coverage mismatch: iteration={len(iteration_pngs)}, recall={len(recall_pngs)}")
    for path in [*iteration_pngs, *recall_pngs]:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("zero dimensions")
        except Exception as exc:
            raise ValueError(f"report figure is unreadable: {path}: {exc}") from exc
        passed("figure", str(path.relative_to(root)), f"bytes={path.stat().st_size}")

    report = pd.DataFrame(checks, columns=["category", "item", "passed", "detail"])
    report.to_csv(root / "verification_report.csv", index=False, encoding="utf-8-sig")
    summary = (
        "# 独立验证摘要\n\n"
        f"- 结论：**PASS**\n- 通过检查：{len(report)}\n- 模型运行：12\n"
        f"- 逐类 Recall 差值：72 行\n- 模型重载概率容差：1e-12\n"
        "- 诊断键连接：6 组均 unmatched=0\n"
    )
    (root / "verification_summary.md").write_text(summary, encoding="utf-8")
    return report
