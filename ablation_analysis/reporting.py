"""Deterministic reporting and decision rules for the paired ablation study."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import FontProperties, findSystemFonts

from .config import LABEL_ORDER


CHANNELS = (3, 4, 5)
SPLIT_MODES = ("record", "temporal")
FEATURE_SETS = ("features_43", "features_40")
LEVELS = ("window", "record")
CHINESE_FONT = Path("/System/Library/Fonts/STHeiti Medium.ttc")
REQUIRED_CJK_GLYPHS = "松动轴承故障真实类别预测"
CJK_CANDIDATE_PATHS = (
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)

_DELTA_REQUIRED = [
    "channel",
    "split_mode",
    "record_macro_f1_delta",
    "looseness_recall_delta",
    "bearing_recall_delta",
]


def _validate_six_runs(frame: pd.DataFrame, numeric_columns) -> None:
    missing = [column for column in ["channel", "split_mode", *numeric_columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(frame) != 6 or frame.duplicated(["channel", "split_mode"]).any():
        raise ValueError("expected exactly six unique channel/split rows")
    actual = set(zip(frame["channel"], frame["split_mode"]))
    expected = {(channel, split) for channel in CHANNELS for split in SPLIT_MODES}
    if actual != expected:
        raise ValueError("channel/split coverage must be exactly CH3/CH4/CH5 x record/temporal")
    try:
        values = frame[list(numeric_columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("guardrail values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("guardrail values must be finite")


def recommend_feature_set(deltas: pd.DataFrame) -> dict:
    """Apply the pre-registered record-level 40-versus-43 guardrails."""
    numeric = _DELTA_REQUIRED[2:]
    _validate_six_runs(deltas, numeric)
    limits = {
        "record_macro_f1_delta": -0.005,
        "looseness_recall_delta": -0.01,
        "bearing_recall_delta": -0.01,
    }
    failures = []
    ordered = deltas.sort_values(["channel", "split_mode"], kind="stable")
    for row in ordered.itertuples(index=False):
        for metric, minimum in limits.items():
            value = float(getattr(row, metric))
            if value < minimum:
                failures.append(
                    {
                        "channel": int(row.channel),
                        "split_mode": row.split_mode,
                        "metric": metric,
                        "value": value,
                        "minimum": minimum,
                    }
                )
    passed = not failures
    return {
        "recommended_feature_set": "features_40" if passed else "features_43",
        "passed": passed,
        "failed_guardrails": failures,
    }


def assess_ch5_unique_value(recalls: pd.DataFrame) -> dict:
    """Judge CH5 class value only when its strict lead holds in both splits."""
    required = ["channel", "split_mode", "class", "recall"]
    missing = [column for column in required if column not in recalls.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    expected = {(c, s, label) for c in CHANNELS for s in SPLIT_MODES for label in LABEL_ORDER}
    actual = set(zip(recalls["channel"], recalls["split_mode"], recalls["class"]))
    if len(recalls) != len(expected) or recalls.duplicated(required[:3]).any() or actual != expected:
        raise ValueError("recall coverage must be exactly channel/split/class")
    try:
        values = recalls["recall"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("recall must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("recall must be finite")

    rows = []
    for label in LABEL_ORDER:
        margins = {}
        for split in SPLIT_MODES:
            subset = recalls[(recalls["class"] == label) & (recalls["split_mode"] == split)]
            ch5 = float(subset.loc[subset["channel"] == 5, "recall"].iloc[0])
            comparator = float(subset.loc[subset["channel"].isin([3, 4]), "recall"].max())
            margins[split] = ch5 - comparator
        unique = margins["record"] > 0 and margins["temporal"] > 0
        rows.append(
            {
                "class": label,
                "record_margin_ch5_vs_best_ch3_ch4": margins["record"],
                "temporal_margin_ch5_vs_best_ch3_ch4": margins["temporal"],
                "direction_consistent": bool((margins["record"] > 0) == (margins["temporal"] > 0)),
                "unique_value": bool(unique),
            }
        )
    table = pd.DataFrame(rows)
    return {"table": table, "unique_classes": table.loc[table["unique_value"], "class"].tolist()}


def recommend_new_features(confusion_evidence: pd.DataFrame, effects: pd.DataFrame) -> dict:
    """Apply the conservative conjunction for proposing a later feature study.

    A directional error is principal when it occurs and is at least as frequent
    as the largest competing off-diagonal error for that actual class. Every
    channel/split/scope must pass in both directions. Weak effects require exact
    43-feature, non-small-sample coverage and a strict majority below 0.147 for
    each directional comparison. No causal inference is made.
    """
    count_columns = {
        "looseness_to_bearing_count",
        "looseness_other_max_offdiag_count",
        "bearing_to_looseness_count",
        "bearing_other_max_offdiag_count",
    }
    required = {"channel", "split_mode", "scope", *count_columns}
    missing = sorted(required - set(confusion_evidence.columns))
    if missing:
        raise ValueError(f"confusion evidence missing required columns: {missing}")
    scopes = {"train_internal_cv", "test_window", "test_record"}
    expected = {(c, split, scope) for c in CHANNELS for split in SPLIT_MODES for scope in scopes}
    if (
        len(confusion_evidence) != len(expected)
        or confusion_evidence.duplicated(["channel", "split_mode", "scope"]).any()
        or set(zip(confusion_evidence["channel"], confusion_evidence["split_mode"], confusion_evidence["scope"])) != expected
    ):
        raise ValueError("confusion evidence must cover exactly channel x split x three scopes")
    numeric = confusion_evidence[list(count_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("confusion counts must be finite and nonnegative")
    evidence = confusion_evidence.copy()
    evidence["looseness_to_bearing_principal"] = (
        evidence["looseness_to_bearing_count"].gt(0)
        & evidence["looseness_to_bearing_count"].ge(evidence["looseness_other_max_offdiag_count"])
    )
    evidence["bearing_to_looseness_principal"] = (
        evidence["bearing_to_looseness_count"].gt(0)
        & evidence["bearing_to_looseness_count"].ge(evidence["bearing_other_max_offdiag_count"])
    )
    evidence["bidirectional_principal"] = evidence[
        ["looseness_to_bearing_principal", "bearing_to_looseness_principal"]
    ].all(axis=1)
    failed = [
        f"CH{row.channel}/{row.split_mode}/{row.scope}"
        for row in evidence.loc[~evidence["bidirectional_principal"]].itertuples(index=False)
    ]

    effect_required = {"channel", "split_mode", "feature", "group_a", "group_b", "abs_delta", "small_sample"}
    missing_effect = sorted(effect_required - set(effects.columns))
    if missing_effect:
        raise ValueError(f"diagnostic effects missing required columns: {missing_effect}")
    try:
        effect_values = effects["abs_delta"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("diagnostic abs_delta must be numeric") from exc
    if not np.isfinite(effect_values).all() or (effect_values < 0).any():
        raise ValueError("diagnostic abs_delta must be finite and nonnegative")
    comparisons = {
        ("correct_looseness", "looseness_to_bearing"),
        ("correct_bearing", "bearing_to_looseness"),
    }
    relevant = effects[
        effects[["group_a", "group_b"]].apply(tuple, axis=1).isin(comparisons)
    ].copy()
    effect_rows = []
    for channel in CHANNELS:
        for split in SPLIT_MODES:
            for group_a, group_b in sorted(comparisons):
                subset = relevant[
                    (relevant.channel == channel) & (relevant.split_mode == split)
                    & (relevant.group_a == group_a) & (relevant.group_b == group_b)
                ]
                coverage_ok = len(subset) == 43 and subset["feature"].nunique() == 43
                sample_ok = coverage_ok and not subset["small_sample"].astype(bool).any()
                weak_share = float(subset["abs_delta"].lt(0.147).mean()) if coverage_ok else 0.0
                weak_majority = sample_ok and weak_share > 0.5
                effect_rows.append({
                    "channel": channel, "split_mode": split, "group_a": group_a, "group_b": group_b,
                    "feature_count": len(subset), "coverage_ok": coverage_ok, "sample_ok": sample_ok,
                    "weak_effect_share": weak_share, "weak_effect_majority": weak_majority,
                })
                if not weak_majority:
                    failed.append(f"CH{channel}/{split}/{group_a}_vs_{group_b}/weak_effects")
    effect_evidence = pd.DataFrame(effect_rows)
    return {
        "recommend_new_features": not failed,
        "failed_conditions": failed,
        "confusion_evidence": evidence,
        "effect_evidence": effect_evidence,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_chinese_font():
    requested = CHINESE_FONT
    required_codepoints = {ord(character) for character in REQUIRED_CJK_GLYPHS}

    def supports_required_glyphs(path: Path) -> bool:
        try:
            available = set(FT2Font(str(path)).get_charmap())
        except (OSError, RuntimeError, ValueError):
            return False
        return required_codepoints.issubset(available)

    preferred_exists = requested.is_file()
    preferred_glyph_check = preferred_exists and supports_required_glyphs(requested)
    if preferred_glyph_check:
        resolved = requested
        fallback = False
    else:
        system_paths = [Path(path) for path in findSystemFonts()]
        candidates = sorted(
            {path for path in (*CJK_CANDIDATE_PATHS, *system_paths) if path.is_file() and path != requested},
            key=lambda path: str(path),
        )
        resolved = next((path for path in candidates if supports_required_glyphs(path)), None)
        if resolved is None:
            raise RuntimeError(
                "No CJK-capable font contains all required Chinese glyphs; "
                f"preferred={requested}, candidates_checked={len(candidates)}"
            )
        fallback = True
        warnings.warn(
            f"Chinese font unavailable or missing glyphs at {requested}; using verified CJK font {resolved}",
            RuntimeWarning,
            stacklevel=2,
        )
    return FontProperties(fname=str(resolved)), {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "fallback_used": fallback,
        "preferred_exists": preferred_exists,
        "preferred_glyph_check_passed": preferred_glyph_check,
        "glyph_check_passed": True,
        "required_glyphs": REQUIRED_CJK_GLYPHS,
    }


def _directional_from_confusion(confusion, channel, split, scope, feature_set) -> dict:
    matrix = np.asarray(confusion, dtype=float)
    if matrix.shape != (len(LABEL_ORDER), len(LABEL_ORDER)) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("confusion matrix must be finite, nonnegative, and 6x6")
    looseness = LABEL_ORDER.index("松动")
    bearing = LABEL_ORDER.index("轴承故障")

    def values(actual, target, actual_name, direction):
        total = float(matrix[actual].sum())
        count = float(matrix[actual, target])
        other = max(
            [float(matrix[actual, predicted]) for predicted in range(len(LABEL_ORDER)) if predicted not in {actual, target}],
            default=0.0,
        )
        return {
            f"{actual_name}_actual_count": total,
            f"{direction}_count": count,
            f"{direction}_rate": count / total if total else 0.0,
            f"{actual_name}_other_max_offdiag_count": other,
        }

    return {
        "channel": channel, "split_mode": split, "scope": scope, "feature_set": feature_set,
        **values(looseness, bearing, "looseness", "looseness_to_bearing"),
        **values(bearing, looseness, "bearing", "bearing_to_looseness"),
    }


def _aggregate_selected_cv_confusion(selected_folds, channel, split, feature_set) -> dict:
    """Aggregate complete OOF pair counts, then find the largest competing pair."""
    offdiag = [
        f"confusion_true_{actual}_pred_{predicted}_count"
        for actual in range(len(LABEL_ORDER))
        for predicted in range(len(LABEL_ORDER))
        if actual != predicted
    ]
    required = {"looseness_actual_count", "bearing_actual_count", *offdiag}
    missing = sorted(required - set(selected_folds.columns))
    if missing or selected_folds.empty:
        raise ValueError(f"selected CV complete confusion evidence missing columns: {missing}")
    values = selected_folds[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("selected CV confusion counts must be finite and nonnegative")
    totals = selected_folds[offdiag].sum()
    looseness = LABEL_ORDER.index("松动")
    bearing = LABEL_ORDER.index("轴承故障")

    def direction(actual, target, actual_name, direction_name):
        target_count = float(totals[f"confusion_true_{actual}_pred_{target}_count"])
        competitors = [
            float(totals[f"confusion_true_{actual}_pred_{predicted}_count"])
            for predicted in range(len(LABEL_ORDER))
            if predicted not in {actual, target}
        ]
        actual_count = float(selected_folds[f"{actual_name}_actual_count"].sum())
        return {
            f"{actual_name}_actual_count": actual_count,
            f"{direction_name}_count": target_count,
            f"{direction_name}_rate": target_count / actual_count if actual_count else 0.0,
            f"{actual_name}_other_max_offdiag_count": max(competitors, default=0.0),
        }

    return {
        "channel": channel, "split_mode": split, "scope": "train_internal_cv", "feature_set": feature_set,
        **direction(looseness, bearing, "looseness", "looseness_to_bearing"),
        **direction(bearing, looseness, "bearing", "bearing_to_looseness"),
    }


def _load_model_artifacts(staging: Path):
    metric_rows = []
    recall_rows = []
    curve_rows = []
    confusion_rows = []
    for channel in CHANNELS:
        for split in SPLIT_MODES:
            for feature_set in FEATURE_SETS:
                run = staging / "models" / split / f"ch{channel}" / feature_set
                metadata = _read_json(run / "metadata.json")
                curves = pd.read_csv(run / "internal_cv_iteration_summary.csv")
                needed = {"iteration", "mean_record_macro_f1"}
                if not needed.issubset(curves.columns):
                    raise ValueError(f"iteration summary missing columns in {run}")
                selected = int(metadata["selected_iteration"])
                selected_rows = curves.loc[curves["iteration"].eq(selected), "mean_record_macro_f1"]
                if len(selected_rows) != 1:
                    raise ValueError(f"selected iteration must occur exactly once in {run}")
                curve = curves.copy()
                curve.insert(0, "feature_set", feature_set)
                curve.insert(0, "split_mode", split)
                curve.insert(0, "channel", channel)
                curve_rows.append(curve)
                window = _read_json(run / "window_metrics.json")
                record = _read_json(run / "record_metrics.json")
                metric_rows.append(
                    {
                        "channel": channel,
                        "split_mode": split,
                        "feature_set": feature_set,
                        "selected_iteration": selected,
                        "internal_cv_selected_record_macro_f1": float(selected_rows.iloc[0]),
                        "test_window_accuracy": float(window["accuracy"]),
                        "test_window_macro_f1": float(window["macro_f1"]),
                        "test_window_weighted_f1": float(window["weighted_f1"]),
                        "test_record_accuracy": float(record["accuracy"]),
                        "test_record_macro_f1": float(record["macro_f1"]),
                        "test_record_weighted_f1": float(record["weighted_f1"]),
                    }
                )
                for level, metrics in (("window", window), ("record", record)):
                    report = metrics["classification_report"]
                    for label in LABEL_ORDER:
                        recall_rows.append(
                            {
                                "channel": channel,
                                "split_mode": split,
                                "feature_set": feature_set,
                                "evaluation_level": level,
                                "class": label,
                                "recall": float(report[label]["recall"]),
                            }
                        )
                fold_scores = pd.read_csv(run / "internal_cv_fold_scores.csv")
                selected_folds = fold_scores[fold_scores["iteration"].eq(selected)]
                confusion_rows.append(
                    _aggregate_selected_cv_confusion(selected_folds, channel, split, feature_set)
                )
                confusion_rows.append(_directional_from_confusion(
                    window["confusion_matrix"], channel, split, "test_window", feature_set
                ))
                confusion_rows.append(_directional_from_confusion(
                    record["confusion_matrix"], channel, split, "test_record", feature_set
                ))
    return (
        pd.DataFrame(metric_rows), pd.DataFrame(recall_rows), pd.concat(curve_rows, ignore_index=True),
        pd.DataFrame(confusion_rows),
    )


def _build_deltas(metrics: pd.DataFrame, recalls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        for split in SPLIT_MODES:
            subset = metrics[(metrics.channel == channel) & (metrics.split_mode == split)].set_index("feature_set")
            row = {"channel": channel, "split_mode": split}
            for column in (
                "internal_cv_selected_record_macro_f1",
                "test_window_accuracy",
                "test_window_macro_f1",
                "test_window_weighted_f1",
                "test_record_accuracy",
                "test_record_macro_f1",
                "test_record_weighted_f1",
            ):
                row[f"{column}_delta"] = float(subset.loc["features_40", column] - subset.loc["features_43", column])
            record_recalls = recalls[
                (recalls.channel == channel)
                & (recalls.split_mode == split)
                & (recalls.evaluation_level == "record")
            ].set_index(["feature_set", "class"])
            row["record_macro_f1_delta"] = row.pop("test_record_macro_f1_delta")
            row["looseness_recall_delta"] = float(
                record_recalls.loc[("features_40", "松动"), "recall"]
                - record_recalls.loc[("features_43", "松动"), "recall"]
            )
            row["bearing_recall_delta"] = float(
                record_recalls.loc[("features_40", "轴承故障"), "recall"]
                - record_recalls.loc[("features_43", "轴承故障"), "recall"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_class_recall_deltas(recalls: pd.DataFrame) -> pd.DataFrame:
    wide = recalls.pivot(
        index=["channel", "split_mode", "evaluation_level", "class"],
        columns="feature_set", values="recall",
    ).reset_index()
    if len(wide) != len(CHANNELS) * len(SPLIT_MODES) * len(LEVELS) * len(LABEL_ORDER):
        raise ValueError("class recall delta coverage is incomplete")
    wide["recall_delta_40_minus_43"] = wide["features_40"] - wide["features_43"]
    return wide[[
        "channel", "split_mode", "evaluation_level", "class",
        "features_43", "features_40", "recall_delta_40_minus_43",
    ]]


def _aggregate_diagnostics(diagnostic_tables: Mapping):
    grouped_rows, effect_rows, error_rows, concentration_rows = [], [], [], []
    for (split, channel), (grouped, effects, errors, summary) in sorted(diagnostic_tables.items()):
        for source, destination in ((grouped, grouped_rows), (effects, effect_rows), (errors, error_rows)):
            enriched = source.copy()
            enriched.insert(0, "channel", channel)
            enriched.insert(1, "split_mode", split)
            destination.append(enriched)
        concentration_rows.append({"channel": channel, "split_mode": split, **summary})
    return (
        pd.concat(grouped_rows, ignore_index=True),
        pd.concat(effect_rows, ignore_index=True),
        pd.concat(error_rows, ignore_index=True),
        pd.DataFrame(concentration_rows),
    )


def _plot_iteration_curves(curves: pd.DataFrame, output: Path, font: FontProperties) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for (channel, split, feature_set), frame in curves.groupby(["channel", "split_mode", "feature_set"], sort=True):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.plot(frame["iteration"], frame["mean_record_macro_f1"], marker="o", linewidth=2)
        ax.set_title(f"CH{channel} {split} {feature_set} 训练内部CV轮数曲线", fontproperties=font)
        ax.set_xlabel("迭代轮数", fontproperties=font)
        ax.set_ylabel("record级 Macro-F1", fontproperties=font)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"ch{channel}_{split}_{feature_set}.png")
        plt.close(fig)


def _plot_class_recalls(recalls: pd.DataFrame, output: Path, font: FontProperties) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_MODES:
        for level in LEVELS:
            frame = recalls[(recalls.split_mode == split) & (recalls.evaluation_level == level)]
            pivot = frame.pivot_table(index="class", columns=["feature_set", "channel"], values="recall")
            pivot = pivot.reindex(LABEL_ORDER)
            fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
            pivot.plot(kind="bar", ax=ax, width=0.85)
            ax.set_title(f"{split} 独立测试{level}级各类Recall", fontproperties=font)
            ax.set_xlabel("真实类别", fontproperties=font)
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1.05)
            ax.set_xticklabels(LABEL_ORDER, rotation=0, fontproperties=font)
            ax.legend(fontsize=8, ncol=3)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / f"{split}_{level}.png")
            plt.close(fig)


def _format_failures(decision: dict) -> str:
    if decision["passed"]:
        return "六组均满足 record 级 Macro-F1、松动 Recall 与轴承故障 Recall 的预注册下限。"
    return "；".join(
        f"CH{item['channel']} {item['split_mode']} {item['metric']}={item['value']:.6f} < {item['minimum']:.6f}"
        for item in decision["failed_guardrails"]
    )


def _write_conclusion(
    path: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    chosen_recalls: pd.DataFrame,
    unique: dict,
    decision: dict,
    effects: pd.DataFrame,
    concentration: pd.DataFrame,
    new_feature_decision: dict,
) -> None:
    chosen = decision["recommended_feature_set"]
    selected = metrics[metrics.feature_set == chosen]
    iteration_lines = "\n".join(
        f"- CH{row.channel} / {row.split_mode}: {int(row.selected_iteration)} 轮，CV record级 Macro-F1={row.internal_cv_selected_record_macro_f1:.4f}"
        for row in selected.itertuples(index=False)
    )
    window_lines = "\n".join(
        f"- CH{row.channel} / {row.split_mode}: Accuracy={row.test_window_accuracy:.4f}，Macro-F1={row.test_window_macro_f1:.4f}，Weighted-F1={row.test_window_weighted_f1:.4f}"
        for row in selected.itertuples(index=False)
    )
    record_lines = "\n".join(
        f"- CH{row.channel} / {row.split_mode}: Accuracy={row.test_record_accuracy:.4f}，Macro-F1={row.test_record_macro_f1:.4f}，Weighted-F1={row.test_record_weighted_f1:.4f}"
        for row in selected.itertuples(index=False)
    )
    recall_lines = "\n".join(
        f"- CH{row.channel} / {row.split_mode} / {row.evaluation_level} / {row['class']}: Recall={row.recall:.4f}"
        for _, row in chosen_recalls.iterrows()
    )
    weak_share = float(new_feature_decision["effect_evidence"]["weak_effect_share"].mean())
    top5_mean = float(concentration["top5_share"].mean()) if len(concentration) else float("nan")
    unique_text = "、".join(unique["unique_classes"]) if unique["unique_classes"] else "无"
    delta_lines = "\n".join(
        f"- CH{row.channel} / {row.split_mode}: record Macro-F1={row.record_macro_f1_delta:+.4f}，松动Recall={row.looseness_recall_delta:+.4f}，轴承故障Recall={row.bearing_recall_delta:+.4f}"
        for row in deltas.itertuples(index=False)
    )
    if new_feature_decision["recommend_new_features"]:
        new_feature_text = "三层双向主要混淆与弱效应条件均满足，**建议开展新特征研究**。"
    else:
        failures = "；".join(new_feature_decision["failed_conditions"])
        new_feature_text = f"严格合取条件未全部满足，**不建议仅据现有证据新增特征**。未满足项：{failures}。"
    text = f"""# 43维与40维去冗余消融结论

## 训练内部 CV（仅用于轮数选择）

{iteration_lines}

这些分数来自训练集内部分组交叉验证，不是独立测试性能。

## 独立测试窗口级

{window_lines}

## 独立测试 record 级

{record_lines}

## 43维 vs 40维与守门决策

所有差值均定义为 `features_40 - features_43`，且松动/轴承守门使用 record 级 Recall。推荐 **{chosen}**。{_format_failures(decision)}
{delta_lines}
更完整的六组差值明细见 `comparison/ablation_deltas.csv`。

## 推荐特征集的 CH3/CH4/CH5 分类 Recall

{recall_lines}

## CH5 独特价值判断

CH5 只在 record 和 temporal 两种划分上都严格高于 CH3、CH4 时，才对该类判为具有独特价值。本次通过的类别：**{unique_text}**。逐类边际见 `comparison/ch5_unique_value.csv`。

## 松动/轴承故障效应量、集中度与是否需要新特征

定向 Cliff's delta 中可忽略效应的比例为 {weak_share:.1%}；六组定向错分的前5个record平均集中度为 {top5_mean:.1%}。
保守规则要求：在 CH3/CH4/CH5 与 record/temporal 六组中，松动→轴承故障与轴承故障→松动都必须至少出现一次，且其计数不低于该真实类别流向任一其他错误类别的最大计数；该条件必须同时出现在训练内部 CV 选定轮数、独立测试窗口级和独立测试 record 级。同时，两个定向比较在每组均必须具备 43 特征完整、非小样本覆盖，且严格多数 `|Cliff's delta| < 0.147`。{new_feature_text}
机器可读决策与逐组证据见 `comparison/new_feature_decision.csv` 和 `comparison/new_feature_evidence.csv`。

## 解释边界

本报告是关联性的消融与错分诊断，**不作因果解释**，不将单一特征偏移解释为故障成因。
"""
    path.write_text(text, encoding="utf-8")


def generate_reports(
    staging_root,
    diagnostic_tables: Mapping,
    consolidated_pairs: pd.DataFrame,
    feature_decisions: pd.DataFrame,
) -> dict:
    """Generate all decision tables, figures, and the bounded Chinese conclusion."""
    staging = Path(staging_root)
    diagnostics_dir = staging / "diagnostics"
    redundancy_dir = staging / "redundancy"
    iteration_dir = staging / "iteration_selection"
    comparison_dir = staging / "comparison"
    for directory in (diagnostics_dir, redundancy_dir, iteration_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    grouped, effects, errors, concentration = _aggregate_diagnostics(diagnostic_tables)
    grouped.to_csv(diagnostics_dir / "four_group_feature_statistics.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(diagnostics_dir / "record_level_cliffs_delta.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(diagnostics_dir / "targeted_error_records.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(diagnostics_dir / "error_concentration.csv", index=False, encoding="utf-8-sig")
    consolidated_pairs.to_csv(
        redundancy_dir / "consolidated_high_correlation_pairs.csv", index=False, encoding="utf-8-sig"
    )
    feature_decisions.to_csv(
        redundancy_dir / "feature_decisions_43_to_40.csv", index=False, encoding="utf-8-sig"
    )

    metrics, recalls, curves, confusion_evidence = _load_model_artifacts(staging)
    deltas = _build_deltas(metrics, recalls)
    class_recall_deltas = _build_class_recall_deltas(recalls)
    decision = recommend_feature_set(deltas[_DELTA_REQUIRED])
    chosen_recalls = recalls[recalls.feature_set.eq(decision["recommended_feature_set"])].copy()
    record_recalls = chosen_recalls[chosen_recalls.evaluation_level.eq("record")][
        ["channel", "split_mode", "class", "recall"]
    ]
    unique = assess_ch5_unique_value(record_recalls)
    chosen_confusion = confusion_evidence[
        confusion_evidence.feature_set.eq(decision["recommended_feature_set"])
    ].drop(columns="feature_set")
    new_feature_decision = recommend_new_features(chosen_confusion, effects)

    curves.to_csv(iteration_dir / "all_iteration_curves.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(comparison_dir / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(comparison_dir / "ablation_deltas.csv", index=False, encoding="utf-8-sig")
    recalls.to_csv(comparison_dir / "channel_class_recall.csv", index=False, encoding="utf-8-sig")
    class_recall_deltas.to_csv(comparison_dir / "class_recall_deltas.csv", index=False, encoding="utf-8-sig")
    unique["table"].to_csv(comparison_dir / "ch5_unique_value.csv", index=False, encoding="utf-8-sig")
    evidence = new_feature_decision["confusion_evidence"].copy()
    evidence["evidence_type"] = "confusion"
    effect_evidence = new_feature_decision["effect_evidence"].copy()
    effect_evidence["evidence_type"] = "effect_size"
    pd.concat([evidence, effect_evidence], ignore_index=True, sort=False).to_csv(
        comparison_dir / "new_feature_evidence.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        "recommend_new_features": new_feature_decision["recommend_new_features"],
        "recommended_feature_set": decision["recommended_feature_set"],
        "rule": "all 18 scopes bidirectionally principal AND all 12 effect comparisons complete, non-small-sample, weak-majority",
        "failed_conditions_json": json.dumps(new_feature_decision["failed_conditions"], ensure_ascii=False),
        "confusion_evidence_json": new_feature_decision["confusion_evidence"].to_json(orient="records", force_ascii=False),
        "effect_evidence_json": new_feature_decision["effect_evidence"].to_json(orient="records", force_ascii=False),
    }]).to_csv(comparison_dir / "new_feature_decision.csv", index=False, encoding="utf-8-sig")
    font, font_metadata = _resolve_chinese_font()
    figures_dir = staging / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "font_metadata.json").write_text(
        json.dumps(font_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_iteration_curves(curves, figures_dir / "iteration_curves", font)
    _plot_class_recalls(recalls, figures_dir / "class_recall", font)
    _write_conclusion(
        staging / "conclusion.md", metrics, deltas, chosen_recalls, unique, decision, effects, concentration,
        new_feature_decision,
    )
    return {"decision": decision, "ch5_unique_value": unique, "new_feature_decision": new_feature_decision}


REPORT_REQUIRED_FILES = (
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
)
