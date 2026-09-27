from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_weight_study.contracts import LABELS
from fusion_weight_study.experiment import (
    ExperimentError,
    channel_weight_source_metrics,
    cross_fitted_fusion,
    derive_final_weights,
    select_strategy,
)


def _oof() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for sample in range(6):
            label = LABELS[sample]
            for channel in (3, 4, 5):
                probs = np.full(6, 0.02)
                winner = sample if channel == (fold + 2) else (sample + channel) % 6
                probs[winner] = 0.9; probs /= probs.sum()
                row = {"record_id": f"f{fold}::{sample}", "window_id": 0,
                       "start_sample": 0, "end_sample": 10, "parent_group_id": f"f{fold}",
                       "label": label, "channel": channel, "fold": fold}
                row.update({f"prob_{name}": probs[i] for i, name in enumerate(LABELS)})
                rows.append(row)
    return pd.DataFrame(rows)


def test_each_outer_fold_weight_provenance_excludes_that_fold() -> None:
    _, provenance = cross_fitted_fusion(_oof())
    for fold in (1, 2, 3):
        used = provenance.loc[provenance["outer_fold"].eq(fold), "source_folds"].unique()
        assert len(used) == 1
        assert str(fold) not in used[0].split(",")


def test_final_weights_come_from_all_oof_rows() -> None:
    weights = derive_final_weights(_oof())
    assert set(weights) == {"global", "classwise"}
    assert sum(weights["global"].values()) == pytest.approx(1.0)
    source = channel_weight_source_metrics(_oof())
    assert set(source["channel"]) == {3, 4, 5}
    assert set(source["metric"]) == {"macro_f1", "class_f1"}
    assert set(source.loc[source["metric"].eq("class_f1"), "label"]) == set(LABELS)


def test_strategy_selection_uses_record_macro_f1_and_simplicity_tolerance() -> None:
    summary = pd.DataFrame([
        {"method": "equal", "record_macro_f1": 0.70, "window_macro_f1": 0.69},
        {"method": "global", "record_macro_f1": 0.704, "window_macro_f1": 0.71},
        {"method": "classwise", "record_macro_f1": 0.703, "window_macro_f1": 0.72},
    ])
    assert select_strategy(summary) == "equal"
    summary.loc[summary.method.eq("global"), "record_macro_f1"] = 0.72
    assert select_strategy(summary) == "global"
    with pytest.raises(ExperimentError, match="strict"):
        select_strategy(summary, strict_results=summary)
