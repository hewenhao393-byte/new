import numpy as np
import pandas as pd
import pytest

from formal_model_selection.config import META
from formal_model_selection.cv import _pred_frame, run_candidate_cv
from formal_model_selection.config import FEATURES


def _metadata(rows=2):
    data = {column: [f"{column}_{i}" for i in range(rows)] for column in META}
    data["rpm"] = [740] * rows
    data["start_sample"] = list(range(rows))
    data["end_sample"] = list(range(1, rows + 1))
    return pd.DataFrame(data)


def test_pred_frame_rejects_probabilities_that_do_not_sum_to_one():
    frame = _metadata()
    invalid = np.array([[0.4, 0.4], [0.5, 0.5]])

    with pytest.raises(ValueError, match="probability"):
        _pred_frame(frame, invalid, ["正常", "松动"])


def test_pred_frame_rejects_missing_required_metadata():
    frame = _metadata().drop(columns=["record_id"])
    valid = np.array([[0.4, 0.6], [0.5, 0.5]])

    with pytest.raises(ValueError, match="metadata"):
        _pred_frame(frame, valid, ["正常", "松动"])


def test_candidate_cv_uses_one_shared_manifest_and_only_requested_checkpoints(monkeypatch):
    rows = []
    rng = np.random.default_rng(7)
    for label_index, label in enumerate(("正常", "松动")):
        for record_index in range(4):
            row = {
                "record_id": f"r{label_index}_{record_index}",
                "window_id": 0,
                "split": "train_dev",
                "label": label,
                "motor": "Motor-2",
                "rpm": 740,
                "condition": label,
                "state": str(record_index),
                "severity": "unknown",
                "start_sample": 0,
                "end_sample": 4800,
            }
            row.update(dict(zip(FEATURES, rng.normal(label_index, 0.1, len(FEATURES)))))
            rows.append(row)
    table = pd.DataFrame(rows)
    manifest_rows = []
    for fold in (1, 2):
        for record_id in table.record_id:
            record_index = int(record_id.rsplit("_", 1)[1])
            role = "validation" if record_index % 2 == fold - 1 else "fit"
            manifest_rows.append({"fold": fold, "record_id": record_id, "role": role})
    manifest = pd.DataFrame(manifest_rows)
    monkeypatch.setattr("formal_model_selection.cv.CANDIDATES", {
        "tiny": {"depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3,
                 "random_strength": 1, "rsm": 0.7}
    })
    monkeypatch.setattr("formal_model_selection.cv.CHECKPOINTS", [1, 2])
    monkeypatch.setattr("formal_model_selection.cv.MAX_ITERATIONS", 2)

    result = run_candidate_cv({3: table, 4: table, 5: table}, manifest, "a" * 64)

    assert len(result) == 1 * 3 * 2 * 2
    assert set(result["iteration"]) == {1, 2}
    assert set(result["fold_hash"]) == {"a" * 64}
    assert result["validation_records"].eq(4).all()
