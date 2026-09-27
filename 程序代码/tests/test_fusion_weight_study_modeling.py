from __future__ import annotations

import numpy as np
import pandas as pd

from fusion_weight_study.contracts import FEATURE_NAMES, LABELS, MODEL_PARAMS
from fusion_weight_study.data import AlignedData
from fusion_weight_study.modeling import generate_oof_predictions, generate_strict_predictions


class FakeModel:
    def __init__(self, params, calls):
        self.params, self.calls = params, calls
        self.classes_ = np.array(LABELS[::-1])

    def fit(self, x, y):
        self.calls.append((self.params, x.copy(), list(y)))
        return self

    def predict_proba(self, x):
        row = np.arange(1, 7, dtype=float); row /= row.sum()
        return np.tile(row, (len(x), 1))


def _frame(channel, parents):
    rows = []
    for index, parent in enumerate(parents):
        row = {
            "record_id": f"{parent}::{index}", "window_id": 0, "start_sample": 0,
            "end_sample": 10, "parent_group_id": parent,
            "label": LABELS[index % len(LABELS)], "channel": channel,
        }
        row.update({name: float(index) for name in FEATURE_NAMES}); rows.append(row)
    return pd.DataFrame(rows)


def test_oof_uses_fixed_params_fold_boundaries_and_label_order() -> None:
    calls = []
    factory = lambda params: FakeModel(params, calls)
    train = {channel: _frame(channel, ["p1", "p2", "p3"]) for channel in (3, 4, 5)}
    data = AlignedData(train, {c: _frame(c, ["s1"]) for c in (3, 4, 5)}, {"p1": 1, "p2": 2, "p3": 3}, {}, pd.DataFrame())

    predictions, metadata = generate_oof_predictions(data, model_factory=factory)

    assert len(calls) == 9
    assert all(call[0] == MODEL_PARAMS for call in calls)
    assert all(list(call[1].columns) == list(FEATURE_NAMES) for call in calls)
    assert len(predictions) == 9
    assert predictions.groupby(["channel", "record_id"]).size().eq(1).all()
    # Fake model emits columns in reversed label order; output must restore contract order.
    assert predictions.loc[0, f"prob_{LABELS[0]}"] > predictions.loc[0, f"prob_{LABELS[-1]}"]
    assert set(metadata["fold"]) == {1, 2, 3}


def test_strict_models_fit_all_train_dev_only() -> None:
    calls = []
    factory = lambda params: FakeModel(params, calls)
    train = {channel: _frame(channel, ["p1", "p2"]) for channel in (3, 4, 5)}
    strict = {channel: _frame(channel, ["s1"]) for channel in (3, 4, 5)}
    data = AlignedData(train, strict, {"p1": 1, "p2": 2}, {}, pd.DataFrame())

    predictions, metadata = generate_strict_predictions(data, model_factory=factory)

    assert len(calls) == 3
    assert all(len(call[1]) == 2 for call in calls)
    assert len(predictions) == 3
    assert set(predictions["record_id"]) == {"s1::0"}
    assert metadata["fit_scope"].eq("all_train_dev").all()
