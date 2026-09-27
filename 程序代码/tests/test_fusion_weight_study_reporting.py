from __future__ import annotations

import json

import pandas as pd
import pytest

from fusion_weight_study.contracts import LABELS
from fusion_weight_study.reporting import OutputExistsError, publish_results


def test_publishes_complete_artifact_tree_without_overwrite(tmp_path) -> None:
    output = tmp_path / "result"
    frame = pd.DataFrame([{"method": "equal", "macro_f1": 0.5}])
    matrix = pd.DataFrame([[1 if i == j else 0 for j in range(6)] for i in range(6)], index=LABELS, columns=LABELS)
    publish_results(
        output,
        tables={
            "fusion_summary.csv": frame,
            "train_side_cv/fold_metrics.csv": frame,
            "train_side_cv/per_class_metrics.csv": frame,
            "train_side_cv/disagreement.csv": frame,
            "train_side_cv/predictions.csv": frame,
            "train_side_cv/channel_oof_predictions.csv": frame,
            "train_side_cv/channel_weight_source_metrics.csv": frame,
            "strict_parent_group/metrics.csv": frame,
            "strict_parent_group/per_class_metrics.csv": frame,
            "strict_parent_group/disagreement.csv": frame,
            "strict_parent_group/predictions.csv": frame,
            "overdependence_diagnostics.csv": frame,
        },
        json_files={"channel_weights.json": {"3": 0.3}, "classwise_weights.json": {"3": {LABELS[0]: 0.3}}},
        matrices={("train_side_cv", "equal", "window"): matrix, ("strict_parent_group", "equal", "record"): matrix},
        manifest={"strict_used_for_selection": False},
        readme="# result\n",
    )
    required = [
        "fusion_summary.csv", "channel_weights.json", "classwise_weights.json",
        "train_side_cv/fold_metrics.csv", "train_side_cv/per_class_metrics.csv",
        "train_side_cv/disagreement.csv", "train_side_cv/predictions.csv",
        "train_side_cv/channel_oof_predictions.csv", "train_side_cv/channel_weight_source_metrics.csv",
        "strict_parent_group/metrics.csv", "strict_parent_group/per_class_metrics.csv",
        "strict_parent_group/disagreement.csv", "strict_parent_group/predictions.csv",
        "overdependence_diagnostics.csv", "run_manifest.json", "README.md",
        "train_side_cv/confusion_equal_window.csv", "train_side_cv/confusion_equal_window.png",
    ]
    assert all((output / name).is_file() for name in required)
    assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["strict_used_for_selection"] is False
    with pytest.raises(OutputExistsError):
        publish_results(output, tables={}, json_files={}, matrices={}, manifest={}, readme="")
