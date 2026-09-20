"""Orchestration for paired 43-versus-40 feature ablations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from baseline_analysis.acceptance import accept_feature_tables, write_acceptance

from .config import CV_SPLITS, FEATURE_40, FEATURE_43, ITERATION_GRID, MAX_ITERATIONS
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


def _correlation_path(root: Path) -> Path | None:
    candidates = [
        root / "correlations" / "pearson_high_correlation_pairs.csv",
        root / "pearson_high_correlation_pairs.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _diagnostics(merged: pd.DataFrame, output: Path, features: Sequence[str]) -> None:
    frame = merged.copy()
    frame["target_group"] = assign_target_group(frame["label"], frame["predicted_label"])
    grouped_feature_summary(frame, features).to_csv(
        output / "grouped_feature_summary.csv", index=False, encoding="utf-8-sig"
    )
    available = set(frame["target_group"].dropna())
    required = {"correct_looseness", "looseness_to_bearing", "correct_bearing", "bearing_to_looseness"}
    if required.issubset(available):
        effects = effect_size_table(frame, features)
    else:
        effects = pd.DataFrame(
            columns=["feature", "group_a", "group_b", "n_records_a", "n_records_b", "delta", "abs_delta", "magnitude", "small_sample"]
        )
    effects.to_csv(output / "record_effect_sizes.csv", index=False, encoding="utf-8-sig")
    per_record, summary = targeted_error_concentration(frame)
    per_record.to_csv(output / "targeted_error_records.csv", index=False, encoding="utf-8-sig")
    (output / "targeted_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(
    feature_source_root,
    baseline_root,
    output_root,
    *,
    checkpoints: Sequence[int] = ITERATION_GRID,
    max_iterations: int = MAX_ITERATIONS,
    cv_splits: int = CV_SPLITS,
    acceptance_loader: Callable = accept_feature_tables,
):
    """Run all 12 paired ablations; test rows never enter fold construction or CV."""
    source = Path(feature_source_root)
    baseline = Path(baseline_root)
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")

    acceptance, tables = acceptance_loader(source)
    prediction_paths = {}
    merged_inputs = {}
    for mode in ("record", "temporal"):
        for channel in (3, 4, 5):
            path = _prediction_path(baseline, mode, channel)
            prediction_paths[(mode, channel)] = path
            predictions = pd.read_csv(path, low_memory=False)
            merged_inputs[(mode, channel)] = validate_and_merge_inputs(tables[mode][channel], predictions)

    root.mkdir(parents=True)
    write_acceptance(acceptance, root / "feature_acceptance")
    build_run_matrix().to_csv(root / "run_matrix.csv", index=False, encoding="utf-8-sig")

    redundancy_dir = root / "redundancy"
    redundancy_dir.mkdir()
    redundancy_decisions(FEATURE_43).to_csv(
        redundancy_dir / "feature_decisions.csv", index=False, encoding="utf-8-sig"
    )
    correlation_path = _correlation_path(baseline)
    if correlation_path is None:
        consolidated = pd.DataFrame(
            columns=["feature_a", "feature_b", "occurrence_count", "channels", "split_modes", "min_abs_r", "max_abs_r", "signs"]
        )
    else:
        correlation_rows = pd.read_csv(correlation_path)
        consolidated = consolidate_pairs(correlation_rows)
    consolidated.to_csv(redundancy_dir / "consolidated_high_correlation_pairs.csv", index=False, encoding="utf-8-sig")

    diagnostics_dir = root / "diagnostics"
    diagnostics_dir.mkdir()
    folds_dir = root / "fold_manifests"
    folds_dir.mkdir()
    models_dir = root / "models"
    models_dir.mkdir()
    run_rows = []
    for channel in (3, 4, 5):
        for mode in ("record", "temporal"):
            data = tables[mode][channel]
            training_split = "train_dev" if mode == "record" else "train"
            train = data.loc[data["split"].eq(training_split)].reset_index(drop=True)
            folds = make_record_folds(train["label"], train["record_id"], n_splits=cv_splits)
            manifest, fold_indices, fold_hash = folds
            stem = f"{mode}_ch{channel}"
            manifest.to_csv(folds_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
            arrays = {}
            for fold_number, (fit_index, validation_index) in enumerate(fold_indices, start=1):
                arrays[f"fold_{fold_number}_fit"] = fit_index
                arrays[f"fold_{fold_number}_validation"] = validation_index
            np.savez(folds_dir / f"{stem}_indices.npz", **arrays)

            diagnostic_out = diagnostics_dir / stem
            diagnostic_out.mkdir()
            _diagnostics(merged_inputs[(mode, channel)], diagnostic_out, FEATURE_43)

            for feature_set, features in (("features_43", FEATURE_43), ("features_40", FEATURE_40)):
                selection = select_iterations(
                    train,
                    features,
                    folds,
                    checkpoints=checkpoints,
                    max_iterations=max_iterations,
                )
                run_out = models_dir / mode / f"ch{channel}" / feature_set
                result = train_selected_model(
                    data,
                    mode,
                    channel,
                    features,
                    selection["selected_iteration"],
                    selection["fold_hash"],
                    selection["fold_scores"],
                    selection["summary"],
                    run_out,
                    iteration_grid=checkpoints,
                )
                run_rows.append(
                    {
                        "channel": channel,
                        "split_mode": mode,
                        "feature_set": feature_set,
                        "selected_iteration": selection["selected_iteration"],
                        "fold_sha256": fold_hash,
                        "window_macro_f1": result["window"]["macro_f1"],
                        "record_macro_f1": result["record"]["macro_f1"],
                    }
                )

    results = pd.DataFrame(run_rows)
    results.to_csv(root / "ablation_results.csv", index=False, encoding="utf-8-sig")
    input_paths = [
        source / folder / f"features_ch{channel}.csv"
        for folder in ("file_split", "temporal_split")
        for channel in (3, 4, 5)
    ] + list(prediction_paths.values())
    if (source / "temporal_split" / "temporal_split.csv").is_file():
        input_paths.append(source / "temporal_split" / "temporal_split.csv")
    if correlation_path is not None:
        input_paths.append(correlation_path)
    run_manifest = {
        "feature_source_root": str(source.resolve()),
        "baseline_root": str(baseline.resolve()),
        "runs": 12,
        "channels": [3, 4, 5],
        "split_modes": ["record", "temporal"],
        "feature_sets": {"features_43": FEATURE_43, "features_40": FEATURE_40},
        "iteration_grid": [int(value) for value in checkpoints],
        "max_iterations": int(max_iterations),
        "cv_splits": int(cv_splits),
        "input_sha256": {str(path.resolve()): _sha256(path) for path in input_paths},
    }
    (root / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
