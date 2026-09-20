import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import ITERATION_GRID, LABEL_ORDER
from ablation_analysis.iteration_selection import (
    aggregate_cv_scores,
    choose_iteration,
    make_record_folds,
    select_iterations,
    staged_record_scores,
)


def _fold_inputs(records_per_class=5, windows=2):
    labels = []
    record_ids = []
    for label_index, label in enumerate(LABEL_ORDER):
        for record_index in range(records_per_class):
            record_id = f"c{label_index}-r{record_index}"
            labels.extend([label] * windows)
            record_ids.extend([record_id] * windows)
    return pd.Series(labels), pd.Series(record_ids)


def _validation(feature_names=("f1", "f2")):
    rows = []
    for class_index, label in enumerate(LABEL_ORDER):
        for window in range(2):
            row = {
                "record_id": f"r-{class_index}",
                "label": label,
                "motor": "M",
                "rpm": 740,
                "condition": "C",
                "state": "S",
                "severity": "N",
                "f1": class_index + window / 10,
                "f2": window,
            }
            rows.append(row)
    return pd.DataFrame(rows)


class FakeStagedModel:
    def __init__(self, stages, classes=LABEL_ORDER):
        self._stages = stages
        self.classes_ = np.asarray(classes)
        self.calls = 0
        self.seen_columns = None

    def staged_predict_proba(self, features):
        self.calls += 1
        self.seen_columns = features.columns.tolist()
        yield from self._stages


def _perfect_probabilities(validation):
    probabilities = np.full((len(validation), len(LABEL_ORDER)), 0.01)
    for row_index, label in enumerate(validation["label"]):
        probabilities[row_index, LABEL_ORDER.index(label)] = 0.95
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def test_production_iteration_grid_has_all_40_checkpoints():
    assert ITERATION_GRID == list(range(20, 801, 20))
    assert len(ITERATION_GRID) == 40


def test_record_folds_are_deterministic_disjoint_and_cover_each_row_once():
    labels, record_ids = _fold_inputs()

    manifest, indices, fold_hash = make_record_folds(labels, record_ids)
    manifest2, indices2, fold_hash2 = make_record_folds(labels, record_ids)

    pd.testing.assert_frame_equal(manifest, manifest2)
    assert fold_hash == fold_hash2 == manifest["fold_sha256"].iloc[0]
    validation_counts = np.zeros(len(labels), dtype=int)
    for (fit_idx, validation_idx), (fit_idx2, validation_idx2) in zip(indices, indices2):
        assert np.array_equal(fit_idx, fit_idx2)
        assert np.array_equal(validation_idx, validation_idx2)
        assert set(record_ids.iloc[fit_idx]).isdisjoint(set(record_ids.iloc[validation_idx]))
        validation_counts[validation_idx] += 1
    assert validation_counts.tolist() == [1] * len(labels)
    assert manifest.columns.tolist() == ["fold", "role", "record_id", "fold_sha256"]


def test_record_folds_reject_record_with_multiple_labels():
    labels, record_ids = _fold_inputs()
    labels.iloc[1] = LABEL_ORDER[1]

    with pytest.raises(ValueError, match="one label"):
        make_record_folds(labels, record_ids)


def test_record_folds_reject_insufficient_records_in_a_class():
    labels, record_ids = _fold_inputs(records_per_class=4)

    with pytest.raises(ValueError, match="unique records per class"):
        make_record_folds(labels, record_ids, n_splits=5)


def test_record_folds_reject_misaligned_or_null_inputs():
    labels, record_ids = _fold_inputs()
    with pytest.raises(ValueError, match="aligned lengths"):
        make_record_folds(labels.iloc[:-1], record_ids)
    record_ids.iloc[0] = None
    with pytest.raises(ValueError, match="null"):
        make_record_folds(labels, record_ids)


def test_aggregate_and_choose_iteration_use_mean_then_smallest_exact_tie():
    scores = pd.DataFrame(
        {
            "fold": [1, 2, 1, 2, 1, 2],
            "iteration": [20, 20, 40, 40, 60, 60],
            "record_macro_f1": [0.8, 1.0, 0.9, 0.9, 0.7, 0.8],
        }
    )

    summary = aggregate_cv_scores(scores)

    assert summary["iteration"].tolist() == [20, 40, 60]
    assert summary["fold_count"].tolist() == [2, 2, 2]
    assert summary.loc[0, "mean_record_macro_f1"] == pytest.approx(0.9)
    assert summary.loc[0, "std_record_macro_f1"] == pytest.approx(np.std([0.8, 1.0], ddof=1))
    assert choose_iteration(summary, checkpoints=[20, 40, 60], expected_folds=2) == 20


def test_choose_iteration_rejects_duplicates_non_grid_and_missing_folds():
    valid = pd.DataFrame(
        {
            "iteration": [20, 40],
            "mean_record_macro_f1": [0.8, 0.9],
            "std_record_macro_f1": [0.1, 0.1],
            "fold_count": [2, 2],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        choose_iteration(pd.concat([valid, valid.iloc[[0]]]), checkpoints=[20, 40], expected_folds=2)
    with pytest.raises(ValueError, match="grid"):
        choose_iteration(valid.assign(iteration=[20, 30]), checkpoints=[20, 40], expected_folds=2)
    with pytest.raises(ValueError, match="fold"):
        choose_iteration(valid.assign(fold_count=[2, 1]), checkpoints=[20, 40], expected_folds=2)


def test_staged_record_scores_calls_stream_once_and_preserves_feature_order():
    validation = _validation()
    perfect = _perfect_probabilities(validation)
    weak = np.full_like(perfect, 1 / len(LABEL_ORDER))
    model = FakeStagedModel([weak, perfect, perfect])

    scores = staged_record_scores(
        model,
        validation,
        ["f2", "f1"],
        checkpoints=[1, 3],
        classes=LABEL_ORDER,
    )

    assert model.calls == 1
    assert model.seen_columns == ["f2", "f1"]
    assert scores["iteration"].tolist() == [1, 3]
    assert scores.loc[1, "record_macro_f1"] == pytest.approx(1.0)


def test_staged_record_scores_rejects_incomplete_stream():
    validation = _validation()
    model = FakeStagedModel([_perfect_probabilities(validation)])

    with pytest.raises(ValueError, match="incomplete"):
        staged_record_scores(model, validation, ["f1", "f2"], [1, 2], LABEL_ORDER)


@pytest.mark.parametrize("checkpoints", [[], [0], [2, 1], [1, 1]])
def test_staged_record_scores_rejects_invalid_checkpoints(checkpoints):
    with pytest.raises(ValueError, match="checkpoint"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            _validation(),
            ["f1", "f2"],
            checkpoints,
            LABEL_ORDER,
        )


@pytest.mark.parametrize(
    "stages,classes,match",
    [
        ([np.full((12, 6), np.nan)], LABEL_ORDER, "probabil"),
        ([np.full((12, 6), 1 / 6)], LABEL_ORDER[:-1] + ["unknown"], "classes"),
        ([np.full((12, 5), 1 / 5)], LABEL_ORDER, "shape"),
    ],
)
def test_staged_record_scores_rejects_invalid_probabilities_and_classes(stages, classes, match):
    with pytest.raises(ValueError, match=match):
        staged_record_scores(FakeStagedModel(stages, classes), _validation(), ["f1", "f2"], [1], classes)


def test_staged_record_scores_rejects_unknown_truth_and_missing_metadata():
    validation = _validation()
    validation.loc[0, "label"] = "unknown"
    with pytest.raises(ValueError, match="unknown label"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            validation,
            ["f1", "f2"],
            [1],
            LABEL_ORDER,
        )

    with pytest.raises(ValueError, match="missing"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            _validation().drop(columns="motor"),
            ["f1", "f2"],
            [1],
            LABEL_ORDER,
        )


def test_select_iterations_runs_one_real_catboost_model_per_reusable_fold(monkeypatch):
    import ablation_analysis.iteration_selection as module

    labels, record_ids = _fold_inputs()
    train = pd.DataFrame(
        {
            "record_id": record_ids,
            "label": labels,
            "motor": "M",
            "rpm": 740,
            "condition": "C",
            "state": "S",
            "severity": "N",
        }
    )
    class_number = train["label"].map({label: index for index, label in enumerate(LABEL_ORDER)})
    train["signal"] = class_number + np.tile([0.0, 0.01], len(train) // 2)
    train["constant"] = 1.0
    folds = make_record_folds(train["label"], train["record_id"])
    original_classifier = module.CatBoostClassifier
    models = []

    def tracking_classifier(**params):
        models.append(params)
        return original_classifier(**params)

    monkeypatch.setattr(module, "MAX_ITERATIONS", 2)
    monkeypatch.setattr(module, "CatBoostClassifier", tracking_classifier)

    result = select_iterations(
        train,
        ["signal", "constant"],
        folds[0],
        folds[1],
        checkpoints=[1, 2],
    )

    assert len(models) == 5
    assert all(params["iterations"] == 2 for params in models)
    assert result["fold_hash"] == folds[2]
    assert result["fold_scores"].shape == (10, 3)
    assert result["summary"]["fold_count"].tolist() == [5, 5]
    assert result["selected_iteration"] in [1, 2]


def test_select_iterations_rejects_indices_that_do_not_match_manifest(monkeypatch):
    labels, record_ids = _fold_inputs()
    train = pd.DataFrame(
        {
            "record_id": record_ids,
            "label": labels,
            "motor": "M",
            "rpm": 740,
            "condition": "C",
            "state": "S",
            "severity": "N",
            "f1": np.arange(len(labels), dtype=float),
        }
    )
    manifest, indices, fold_hash = make_record_folds(labels, record_ids)
    tampered = [(fit.copy(), validation.copy()) for fit, validation in indices]
    tampered[0] = (tampered[0][0][1:], tampered[0][1])
    monkeypatch.setattr("ablation_analysis.iteration_selection.MAX_ITERATIONS", 1)

    with pytest.raises(ValueError, match="manifest"):
        select_iterations(train, ["f1"], (manifest, tampered, fold_hash), checkpoints=[1])
