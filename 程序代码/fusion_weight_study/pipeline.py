from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import CHANNELS, FEATURE_NAMES, LABELS, MODEL_PARAMS, validate_source_manifest
from .data import WINDOW_KEY, load_aligned_data, sha256_file
from .experiment import PROBABILITY_COLUMNS, channel_weight_source_metrics, cross_fitted_fusion, derive_final_weights, fuse_long_predictions, select_strategy
from .metrics import aggregate_records, disagreement_statistics, evaluate_predictions, overdependence_rows
from .modeling import generate_oof_predictions


def load_existing_strict_predictions(root: Path | str) -> tuple[pd.DataFrame, dict[str, str]]:
    root = Path(root)
    outputs, hashes = [], {}
    for channel in CHANNELS:
        path = root / "models" / f"ch{channel}" / "window_predictions.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {*WINDOW_KEY, "parent_group_id", "label", "channel", *LABELS}
        if not required.issubset(frame.columns) or not frame["channel"].eq(channel).all():
            raise ValueError(f"invalid existing strict prediction file: {path}")
        selected = frame.loc[:, [*WINDOW_KEY, "parent_group_id", "label", "channel", *LABELS]].copy()
        selected = selected.rename(columns={label: f"prob_{label}" for label in LABELS})
        outputs.append(selected); hashes[str(path)] = sha256_file(path)
    combined = pd.concat(outputs, ignore_index=True)
    # Alignment validation is performed by a no-op equal fusion attempt.
    fuse_long_predictions(combined, "equal")
    return combined, hashes


def _evaluate_all(fused: pd.DataFrame, split: str):
    summaries, classes, matrices = [], [], {}
    for method in ("equal", "global", "classwise"):
        method_frame = fused.loc[fused["method"].eq(method)]
        for level, evaluated in (("window", method_frame), ("record", aggregate_records(method_frame))):
            summary, per_class, matrix = evaluate_predictions(evaluated, level)
            summary.update({"split": split, "method": method}); summaries.append(summary)
            per_class.insert(0, "method", method); per_class.insert(0, "split", split); classes.append(per_class)
            matrices[(split, method, level)] = matrix
    return pd.DataFrame(summaries), pd.concat(classes, ignore_index=True), matrices


def _fold_metrics(fused: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, method), part in fused.groupby(["fold", "method"]):
        for level, evaluated in (("window", part), ("record", aggregate_records(part))):
            summary, _, _ = evaluate_predictions(evaluated, level)
            summary.update({"fold": int(fold), "method": method}); rows.append(summary)
    return pd.DataFrame(rows)


def _selection_table(summary: pd.DataFrame) -> pd.DataFrame:
    train = summary.loc[summary["split"].eq("train_side_cv")]
    return train.pivot(index="method", columns="level", values="macro_f1").reset_index().rename(
        columns={"window": "window_macro_f1", "record": "record_macro_f1"}
    )


def _leave_one_out(raw: pd.DataFrame, selected: str, weights, split: str) -> pd.DataFrame:
    method_weights = None if selected == "equal" else weights[selected]
    full = fuse_long_predictions(raw, selected, method_weights)
    full_records = aggregate_records(full)
    full_summary, _, _ = evaluate_predictions(full_records, "record")
    full_labels = np.asarray(LABELS)[full_records.loc[:, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)]
    rows = []
    for removed in CHANNELS:
        reduced = fuse_long_predictions(raw.loc[~raw["channel"].eq(removed)], selected, method_weights)
        reduced_records = aggregate_records(reduced)
        reduced_summary, _, _ = evaluate_predictions(reduced_records, "record")
        reduced_labels = np.asarray(LABELS)[reduced_records.loc[:, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)]
        drop = full_summary["macro_f1"] - reduced_summary["macro_f1"]
        rows.append({
            "scope": "leave_one_channel_out", "split": split, "method": selected,
            "removed_channel": removed, "full_record_macro_f1": full_summary["macro_f1"],
            "reduced_record_macro_f1": reduced_summary["macro_f1"], "macro_f1_drop": drop,
            "prediction_flip_rate": float(np.mean(full_labels != reduced_labels)),
            "risk_flag": bool(drop > 0.05),
        })
    return pd.DataFrame(rows)


def _git_state(project_root: Path) -> dict[str, object]:
    def run(*args):
        return subprocess.run(args, cwd=project_root, text=True, capture_output=True, check=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def run_study(args) -> Path:
    if args.model_thread_count != MODEL_PARAMS["thread_count"]:
        raise ValueError("model thread count is frozen at 4 by the P1@500 contract")
    source_manifest = validate_source_manifest(args.source_manifest)
    data = load_aligned_data({3: args.ch3, 4: args.ch4, 5: args.ch5}, args.split, args.folds)
    if args.dry_run:
        print(json.dumps({
            "status": "validated", "feature_count": len(FEATURE_NAMES), "labels": list(LABELS),
            "train_windows_per_channel": len(data.train_dev[3]), "strict_windows_per_channel": len(data.strict[3]),
            "train_parents": data.train_dev[3]["parent_group_id"].nunique(),
            "strict_parents": data.strict[3]["parent_group_id"].nunique(), "folds": len(set(data.fold_by_parent.values())),
        }, ensure_ascii=False, indent=2))
        return Path(args.output)

    from .reporting import publish_results

    oof, model_meta = generate_oof_predictions(data)
    cross_fused, provenance = cross_fitted_fusion(oof)
    final_weights = derive_final_weights(oof)
    strict_raw, strict_hashes = load_existing_strict_predictions(Path(args.source_manifest).parent)
    strict_parts = [fuse_long_predictions(strict_raw, method, None if method == "equal" else final_weights[method]) for method in ("equal", "global", "classwise")]
    strict_fused = pd.concat(strict_parts, ignore_index=True)

    train_summary, train_classes, train_matrices = _evaluate_all(cross_fused, "train_side_cv")
    strict_summary, strict_classes, strict_matrices = _evaluate_all(strict_fused, "strict_parent_group")
    summary = pd.concat([train_summary, strict_summary], ignore_index=True)
    selection = _selection_table(summary)
    selected = select_strategy(selection)
    diagnostic = overdependence_rows(final_weights["global"], final_weights["classwise"])
    diagnostic = pd.concat([
        diagnostic,
        _leave_one_out(oof, selected, final_weights, "train_side_cv"),
        _leave_one_out(strict_raw, selected, final_weights, "strict_parent_group"),
    ], ignore_index=True, sort=False)
    disagreements = pd.concat([
        disagreement_statistics(oof, split="train_side_cv", level="window"),
        disagreement_statistics(oof, split="train_side_cv", level="record"),
        disagreement_statistics(strict_raw, split="strict_parent_group", level="window"),
        disagreement_statistics(strict_raw, split="strict_parent_group", level="record"),
    ], ignore_index=True)
    fold_metrics = _fold_metrics(cross_fused)
    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "experiment": "three_channel_fusion_weight_study", "strict_used_for_selection": False,
        "weight_metric": "window_level_grouped_cv_f1", "selected_method": selected,
        "selection_rule": {"primary": "record_macro_f1", "secondary": "window_macro_f1", "simplicity_tolerance": 0.005},
        "labels": list(LABELS), "features": list(FEATURE_NAMES), "params": MODEL_PARAMS,
        "source_manifest": str(args.source_manifest), "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_hashes": {**data.source_hashes, **strict_hashes}, "git": _git_state(project_root),
        "python": sys.version, "platform": platform.platform(), "command": " ".join(sys.argv),
        "weight_provenance": provenance.drop(columns="weights").to_dict(orient="records"),
    }
    readme = (
        "# 三通道融合权重实验\n\n"
        f"- 训练侧 grouped CV 选定方法：`{selected}`\n"
        "- 权重来自训练侧 OOF 六类概率，strict parent_group 未参与权重或策略选择。\n"
        "- strict 结果仅是已冻结策略的独立验证，不代表全量正式部署模型的新泛化声称。\n"
        "- CH3/CH4/CH5 单通道模型参数、43 维特征及预处理未更改。\n"
    )
    return publish_results(
        args.output,
        tables={
            "fusion_summary.csv": summary, "train_side_cv/fold_metrics.csv": fold_metrics,
            "train_side_cv/per_class_metrics.csv": train_classes,
            "train_side_cv/disagreement.csv": disagreements.loc[disagreements["split"].eq("train_side_cv")],
            "train_side_cv/predictions.csv": cross_fused,
            "train_side_cv/channel_oof_predictions.csv": oof,
            "train_side_cv/channel_weight_source_metrics.csv": channel_weight_source_metrics(oof),
            "train_side_cv/channel_model_folds.csv": model_meta,
            "strict_parent_group/metrics.csv": strict_summary,
            "strict_parent_group/per_class_metrics.csv": strict_classes,
            "strict_parent_group/disagreement.csv": disagreements.loc[disagreements["split"].eq("strict_parent_group")],
            "strict_parent_group/predictions.csv": strict_fused,
            "overdependence_diagnostics.csv": diagnostic,
        },
        json_files={"channel_weights.json": final_weights["global"], "classwise_weights.json": final_weights["classwise"]},
        matrices={**train_matrices, **strict_matrices}, manifest=manifest, readme=readme,
    )
