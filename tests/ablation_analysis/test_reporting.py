import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ablation_analysis.config import LABEL_ORDER
from ablation_analysis.reporting import (
    assess_ch5_unique_value,
    generate_reports,
    recommend_feature_set,
    recommend_new_features,
)


def _six_deltas(record=-0.005, looseness=-0.01, bearing=-0.01):
    return pd.DataFrame(
        [
            {
                "channel": channel,
                "split_mode": split_mode,
                "record_macro_f1_delta": record,
                "looseness_recall_delta": looseness,
                "bearing_recall_delta": bearing,
            }
            for channel in (3, 4, 5)
            for split_mode in ("record", "temporal")
        ]
    )


def test_recommend_feature_set_includes_exact_guardrail_boundaries():
    result = recommend_feature_set(_six_deltas())
    assert result["recommended_feature_set"] == "features_40"
    assert result["passed"] is True
    assert result["failed_guardrails"] == []


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("record_macro_f1_delta", -0.0050001),
        ("looseness_recall_delta", -0.0100001),
        ("bearing_recall_delta", -0.0100001),
    ],
)
def test_recommend_feature_set_rejects_any_guardrail_breach(column, value):
    deltas = _six_deltas()
    deltas.loc[0, column] = value
    result = recommend_feature_set(deltas)
    assert result["recommended_feature_set"] == "features_43"
    assert result["passed"] is False
    assert result["failed_guardrails"][0]["metric"] == column


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "missing_column"])
def test_recommend_feature_set_rejects_malformed_six_run_input(mutation):
    deltas = _six_deltas()
    if mutation == "missing":
        deltas = deltas.iloc[:-1]
    elif mutation == "duplicate":
        deltas.loc[5, ["channel", "split_mode"]] = deltas.loc[0, ["channel", "split_mode"]]
    elif mutation == "nonfinite":
        deltas.loc[0, "record_macro_f1_delta"] = np.nan
    else:
        deltas = deltas.drop(columns="bearing_recall_delta")
    with pytest.raises(ValueError):
        recommend_feature_set(deltas)


def _recalls(ch5_record=0.8, ch5_temporal=0.8):
    rows = []
    for split_mode in ("record", "temporal"):
        for channel in (3, 4, 5):
            for label in LABEL_ORDER:
                recall = 0.7
                if channel == 5:
                    recall = ch5_record if split_mode == "record" else ch5_temporal
                rows.append(
                    {"channel": channel, "split_mode": split_mode, "class": label, "recall": recall}
                )
    return pd.DataFrame(rows)


def test_ch5_unique_value_requires_strict_advantage_in_both_splits():
    result = assess_ch5_unique_value(_recalls())
    assert result["unique_classes"] == LABEL_ORDER
    assert result["table"]["unique_value"].all()
    tied = _recalls(ch5_temporal=0.7)
    result = assess_ch5_unique_value(tied)
    assert result["unique_classes"] == []
    assert not result["table"]["unique_value"].any()


def test_ch5_unique_value_validates_exact_coverage():
    malformed = _recalls().iloc[:-1]
    with pytest.raises(ValueError, match="coverage"):
        assess_ch5_unique_value(malformed)


def test_new_features_require_all_three_bidirectional_confusion_scopes_and_weak_effects():
    evidence = pd.DataFrame(
        [
            {"scope": scope, "looseness_to_bearing_principal": True, "bearing_to_looseness_principal": True}
            for scope in ("train_internal_cv", "test_window", "test_record")
        ]
    )
    effects = pd.DataFrame({"magnitude": ["negligible", "small", "medium"]})
    assert recommend_new_features(evidence, effects)["recommend_new_features"] is True
    evidence.loc[0, "bearing_to_looseness_principal"] = False
    result = recommend_new_features(evidence, effects)
    assert result["recommend_new_features"] is False
    assert "train_internal_cv" in result["failed_conditions"]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _synthetic_staging(root: Path):
    diagnostics = {}
    for channel in (3, 4, 5):
        for split_mode in ("record", "temporal"):
            rows = []
            for feature_set, offset in (("features_43", 0.0), ("features_40", -0.001)):
                run = root / "models" / split_mode / f"ch{channel}" / feature_set
                run.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "iteration": [20, 40],
                        "mean_record_macro_f1": [0.70 + offset, 0.72 + offset],
                        "std_record_macro_f1": [0.01, 0.01],
                        "fold_count": [5, 5],
                    }
                ).to_csv(run / "internal_cv_iteration_summary.csv", index=False)
                metadata = {"selected_iteration": 40, "split_mode": split_mode, "channel": channel}
                _write_json(run / "metadata.json", metadata)
                report = {
                    label: {"recall": 0.80 + offset, "precision": 0.8, "f1-score": 0.8, "support": 10}
                    for label in LABEL_ORDER
                }
                metrics = {
                    "accuracy": 0.80 + offset,
                    "macro_f1": 0.80 + offset,
                    "weighted_f1": 0.80 + offset,
                    "classification_report": report,
                }
                _write_json(run / "window_metrics.json", metrics)
                _write_json(run / "record_metrics.json", metrics)
                rows.append({"feature_set": feature_set})
            grouped = pd.DataFrame(
                [{"target_group": "correct_looseness", "feature": "rms", "window_count": 5,
                  "record_count": 5, "median": 1.0, "q1": 0.9, "q3": 1.1, "iqr": 0.2}]
            )
            effects = pd.DataFrame(
                [{"feature": "rms", "group_a": "correct_looseness", "group_b": "looseness_to_bearing",
                  "n_records_a": 5, "n_records_b": 5, "delta": 0.1, "abs_delta": 0.1,
                  "magnitude": "negligible", "small_sample": False}]
            )
            errors = pd.DataFrame(
                [{"target_group": "looseness_to_bearing", "record_id": f"r-{channel}-{split_mode}",
                  "motor": "M", "rpm": 1000, "condition": "c", "state": "s", "severity": "low",
                  "error_windows": 2}]
            )
            diagnostics[(split_mode, channel)] = (grouped, effects, errors, {
                "total_error_windows": 2, "error_records": 1, "top5_error_windows": 2, "top5_share": 1.0
            })
    consolidated = pd.DataFrame([{"feature_a": "rms", "feature_b": "std", "occurrence_count": 6}])
    decisions = pd.DataFrame([{"feature": "std", "decision": "remove", "reason": "redundant"}])
    return diagnostics, consolidated, decisions


def test_generate_reports_writes_exact_tables_sections_and_readable_plots(tmp_path):
    diagnostics, consolidated, decisions = _synthetic_staging(tmp_path)
    generate_reports(tmp_path, diagnostics, consolidated, decisions)
    exact_files = [
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
        "comparison/ch5_unique_value.csv",
        "conclusion.md",
    ]
    for relative in exact_files:
        assert (tmp_path / relative).is_file(), relative
    curves = list((tmp_path / "figures" / "iteration_curves").glob("*.png"))
    assert len(curves) == 12
    recall_plots = list((tmp_path / "figures" / "class_recall").glob("*.png"))
    assert len(recall_plots) >= 4
    for plot in curves + recall_plots:
        with Image.open(plot) as image:
            assert image.width >= 600 and image.height >= 400
    metrics = pd.read_csv(tmp_path / "comparison" / "ablation_metrics.csv")
    assert {
        "internal_cv_selected_record_macro_f1",
        "test_window_macro_f1",
        "test_record_macro_f1",
    }.issubset(metrics.columns)
    conclusion = (tmp_path / "conclusion.md").read_text(encoding="utf-8")
    assert "训练内部 CV" in conclusion
    assert "独立测试窗口级" in conclusion
    assert "独立测试 record 级" in conclusion
    assert "不作因果解释" in conclusion
