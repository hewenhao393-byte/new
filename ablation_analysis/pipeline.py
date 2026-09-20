"""Orchestration for paired 43-versus-40 feature ablations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from baseline_analysis.acceptance import accept_feature_tables, write_acceptance

from .config import CV_SPLITS, FEATURE_40, FEATURE_43, ITERATION_GRID, LABEL_ORDER, MAX_ITERATIONS
from .confusion_diagnostics import (
    assign_target_group,
    effect_size_table,
    grouped_feature_summary,
    targeted_error_concentration,
)
from .input_validation import validate_and_merge_inputs
from .iteration_selection import make_record_folds, select_iterations
from .modeling import train_selected_model
from .redundancy import consolidate_pairs, redundancy_decisions
from .reporting import REPORT_REQUIRED_FILES, generate_reports


_JOIN_KEYS = ["record_id", "window_id", "start_sample", "end_sample"]
_CORRELATION_COLUMNS = ["feature_a", "feature_b", "pearson_r", "channel", "split_mode"]


def build_run_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"channel": channel, "split_mode": mode, "feature_set": feature_set}
            for channel in (3, 4, 5)
            for mode in ("record", "temporal")
            for feature_set in ("features_43", "features_40")
        ],
        columns=["channel", "split_mode", "feature_set"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_path(root: Path, mode: str, channel: int) -> Path:
    candidates = [
        root / "models" / mode / f"ch{channel}" / "window_predictions.csv",
        root / mode / f"ch{channel}" / "window_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"baseline window predictions not found for {mode} ch{channel}")


def _load_correlation_audit(root: Path):
    path = root / "correlations" / "pearson_high_correlation_pairs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required baseline correlation audit not found: {path}")
    rows = pd.read_csv(path)
    missing = [column for column in _CORRELATION_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"baseline correlation audit missing required columns: {missing}")
    if len(rows) != 19:
        raise ValueError(f"baseline correlation audit must contain exactly 19 rows, got {len(rows)}")
    consolidated = consolidate_pairs(rows.loc[:, _CORRELATION_COLUMNS])
    return path, consolidated


def _validate_test_prediction_coverage(features: pd.DataFrame, predictions: pd.DataFrame) -> None:
    test = features.loc[features["split"].eq("test"), _JOIN_KEYS]
    prediction_keys = predictions.loc[:, _JOIN_KEYS]
    if test.duplicated(_JOIN_KEYS).any() or prediction_keys.duplicated(_JOIN_KEYS).any():
        raise ValueError("test features and baseline predictions must have unique JOIN_KEYS")
    expected = set(map(tuple, test.itertuples(index=False, name=None)))
    actual = set(map(tuple, prediction_keys.itertuples(index=False, name=None)))
    if len(test) != len(predictions) or expected != actual:
        raise ValueError(
            "baseline predictions must cover all and only feature rows with split=='test': "
            f"expected={len(test)}, actual={len(predictions)}, missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def _validate_baseline_predictions(predictions: pd.DataFrame) -> None:
    required = ["label", "predicted_label", *LABEL_ORDER]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"baseline predictions missing required columns: {missing}")
    if predictions[["label", "predicted_label"]].isna().any().any():
        raise ValueError("baseline prediction label and predicted_label must be nonnull")
    unknown_actual = sorted(set(predictions["label"]) - set(LABEL_ORDER))
    if unknown_actual:
        raise ValueError(f"baseline predictions contain unknown label values: {unknown_actual}")
    unknown_predicted = sorted(set(predictions["predicted_label"]) - set(LABEL_ORDER))
    if unknown_predicted:
        raise ValueError(f"baseline predictions contain unknown predicted_label values: {unknown_predicted}")
    try:
        probabilities = predictions[LABEL_ORDER].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline class probabilities must be numeric") from exc
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("baseline class probabilities must be finite values in [0, 1] summing to one")
    expected = np.asarray(LABEL_ORDER, dtype=object)[probabilities.argmax(axis=1)]
    if not np.array_equal(expected, predictions["predicted_label"].to_numpy(dtype=object)):
        raise ValueError("predicted_label must match the maximum stored class probability")


def _diagnostic_tables(merged: pd.DataFrame, features: Sequence[str]):
    frame = merged.copy()
    frame["target_group"] = assign_target_group(frame["label"], frame["predicted_label"])
    grouped = grouped_feature_summary(frame, features)
    available = set(frame["target_group"].dropna())
    required = {"correct_looseness", "looseness_to_bearing", "correct_bearing", "bearing_to_looseness"}
    if required.issubset(available):
        effects = effect_size_table(frame, features)
    else:
        effects = pd.DataFrame(
            columns=["feature", "group_a", "group_b", "n_records_a", "n_records_b", "delta", "abs_delta", "magnitude", "small_sample"]
        )
    per_record, summary = targeted_error_concentration(frame)
    return grouped, effects, per_record, summary


def _write_diagnostics(tables, output: Path) -> None:
    grouped, effects, per_record, summary = tables
    grouped.to_csv(output / "grouped_feature_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output / "record_effect_sizes.csv", index=False, encoding="utf-8-sig")
    per_record.to_csv(output / "targeted_error_records.csv", index=False, encoding="utf-8-sig")
    (output / "targeted_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _hash_inputs(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _validate_staging(staging: Path) -> None:
    required_model_files = {
        "model.cbm",
        "metadata.json",
        "internal_cv_fold_scores.csv",
        "internal_cv_iteration_summary.csv",
        "window_predictions.csv",
        "record_predictions.csv",
        "window_metrics.json",
        "record_metrics.json",
        "feature_importance.csv",
    }
    run_dirs = list((staging / "models").glob("*/ch*/features_*"))
    if len(run_dirs) != 12:
        raise RuntimeError(f"staging completeness failure: expected 12 model runs, got {len(run_dirs)}")
    for run_dir in run_dirs:
        missing = required_model_files - {path.name for path in run_dir.iterdir()}
        if missing:
            raise RuntimeError(f"staging completeness failure in {run_dir}: missing {sorted(missing)}")
    manifests = list((staging / "fold_manifests").glob("*.csv"))
    indices = list((staging / "fold_manifests").glob("*_indices.npz"))
    if len(manifests) != 6 or len(indices) != 6:
        raise RuntimeError("staging completeness failure: expected six fold manifests and six index bundles")
    missing_reports = [relative for relative in REPORT_REQUIRED_FILES if not (staging / relative).is_file()]
    if missing_reports:
        raise RuntimeError(f"staging completeness failure: missing report files {missing_reports}")
    iteration_plots = list((staging / "figures" / "iteration_curves").glob("*.png"))
    recall_plots = list((staging / "figures" / "class_recall").glob("*.png"))
    if len(iteration_plots) != 12 or len(recall_plots) < 4:
        raise RuntimeError(
            "staging completeness failure: expected 12 iteration plots and at least 4 class-recall plots"
        )


def run_pipeline(
    feature_source_root,
    baseline_root,
    output_root,
):
    """Run all 12 paired ablations; test rows never enter fold construction or CV."""
    source = Path(feature_source_root)
    baseline = Path(baseline_root)
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")

    prediction_paths = {
        (mode, channel): _prediction_path(baseline, mode, channel)
        for mode in ("record", "temporal")
        for channel in (3, 4, 5)
    }
    correlation_path = baseline / "correlations" / "pearson_high_correlation_pairs.csv"
    source_paths = [
        source / folder / f"features_ch{channel}.csv"
        for folder in ("file_split", "temporal_split")
        for channel in (3, 4, 5)
    ]
    temporal_manifest = source / "temporal_split" / "temporal_split.csv"
    if temporal_manifest.is_file():
        source_paths.append(temporal_manifest)
    input_paths = [*source_paths, *prediction_paths.values(), correlation_path]
    pre_read_hashes = _hash_inputs(input_paths)

    correlation_path, consolidated = _load_correlation_audit(baseline)
    acceptance, tables = accept_feature_tables(source)
    diagnostic_inputs = {}
    fold_inputs = {}
    for mode in ("record", "temporal"):
        for channel in (3, 4, 5):
            path = prediction_paths[(mode, channel)]
            predictions = pd.read_csv(path, low_memory=False)
            _validate_baseline_predictions(predictions)
            _validate_test_prediction_coverage(tables[mode][channel], predictions)
            merged = validate_and_merge_inputs(tables[mode][channel], predictions)
            diagnostic_inputs[(mode, channel)] = _diagnostic_tables(merged, FEATURE_43)
            data = tables[mode][channel]
            training_split = "train_dev" if mode == "record" else "train"
            train = data.loc[data["split"].eq(training_split)].reset_index(drop=True)
            fold_inputs[(mode, channel)] = (
                train,
                make_record_folds(train["label"], train["record_id"], n_splits=CV_SPLITS),
            )

    preflight_hashes = _hash_inputs(input_paths)
    if preflight_hashes != pre_read_hashes:
        raise RuntimeError("authoritative input hash mismatch during preflight; no output was created")

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(f"{root.name}.staging-{uuid4().hex}")
    staging.mkdir()
    try:
        write_acceptance(acceptance, staging / "feature_acceptance")
        build_run_matrix().to_csv(staging / "run_matrix.csv", index=False, encoding="utf-8-sig")

        folds_dir = staging / "fold_manifests"
        folds_dir.mkdir()
        models_dir = staging / "models"
        models_dir.mkdir()
        run_rows = []
        for channel in (3, 4, 5):
            for mode in ("record", "temporal"):
                data = tables[mode][channel]
                train, folds = fold_inputs[(mode, channel)]
                manifest, fold_indices, fold_hash = folds
                stem = f"{mode}_ch{channel}"
                manifest.to_csv(folds_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
                arrays = {}
                for fold_number, (fit_index, validation_index) in enumerate(fold_indices, start=1):
                    arrays[f"fold_{fold_number}_fit"] = fit_index
                    arrays[f"fold_{fold_number}_validation"] = validation_index
                np.savez(folds_dir / f"{stem}_indices.npz", **arrays)

                for feature_set, features in (("features_43", FEATURE_43), ("features_40", FEATURE_40)):
                    selection = select_iterations(
                        train, features, folds, checkpoints=ITERATION_GRID, max_iterations=MAX_ITERATIONS
                    )
                    run_out = models_dir / mode / f"ch{channel}" / feature_set
                    result = train_selected_model(
                        data, mode, channel, features, selection["selected_iteration"],
                        selection["fold_hash"], selection["fold_scores"], selection["summary"], run_out,
                    )
                    run_rows.append(
                        {
                            "channel": channel, "split_mode": mode, "feature_set": feature_set,
                            "selected_iteration": selection["selected_iteration"], "fold_sha256": fold_hash,
                            "window_macro_f1": result["window"]["macro_f1"],
                            "record_macro_f1": result["record"]["macro_f1"],
                        }
                    )

        results = pd.DataFrame(run_rows)
        results.to_csv(staging / "ablation_results.csv", index=False, encoding="utf-8-sig")
        run_manifest = {
            "feature_source_root": str(source.resolve()), "baseline_root": str(baseline.resolve()),
            "runs": 12, "channels": [3, 4, 5], "split_modes": ["record", "temporal"],
            "feature_sets": {"features_43": FEATURE_43, "features_40": FEATURE_40},
            "iteration_grid": [int(value) for value in ITERATION_GRID],
            "max_iterations": int(MAX_ITERATIONS), "cv_splits": int(CV_SPLITS),
            "input_sha256": preflight_hashes,
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        generate_reports(
            staging,
            diagnostic_inputs,
            consolidated,
            redundancy_decisions(FEATURE_43),
        )
        _validate_staging(staging)
        final_hashes = _hash_inputs(input_paths)
        if final_hashes != preflight_hashes:
            changed = sorted(path for path in preflight_hashes if preflight_hashes[path] != final_hashes.get(path))
            raise RuntimeError(f"authoritative input hash mismatch before finalize: {changed}")
        if root.exists():
            raise FileExistsError(f"output directory appeared during run: {root}")
        staging.rename(root)
        return results
    except Exception as exc:
        raise RuntimeError(f"ablation pipeline failed: {exc}; staging retained at: {staging}") from exc
