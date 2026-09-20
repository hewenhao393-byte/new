import numpy as np
import pandas as pd
import pytest

from ablation_analysis.config import FEATURE_40, FEATURE_43, ITERATION_GRID, LABEL_ORDER
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
    assert {
        "looseness_to_bearing_count", "looseness_to_bearing_rate",
        "bearing_to_looseness_count", "bearing_to_looseness_rate",
        "looseness_other_max_offdiag_count", "bearing_other_max_offdiag_count",
    }.issubset(scores.columns)


def test_staged_record_scores_emits_directional_confusion_from_same_stream():
    validation = _validation()
    probabilities = _perfect_probabilities(validation)
    looseness = LABEL_ORDER.index("松动")
    bearing = LABEL_ORDER.index("轴承故障")
    looseness_rows = validation["label"].eq("松动").to_numpy()
    bearing_rows = validation["label"].eq("轴承故障").to_numpy()
    probabilities[looseness_rows] = 0
    probabilities[looseness_rows, bearing] = 1
    probabilities[bearing_rows] = 0
    probabilities[bearing_rows, looseness] = 1
    model = FakeStagedModel([probabilities])

    scores = staged_record_scores(model, validation, ["f1", "f2"], [1], LABEL_ORDER)

    assert model.calls == 1
    assert scores.loc[0, "looseness_to_bearing_count"] == 1
    assert scores.loc[0, "bearing_to_looseness_count"] == 1
    assert scores.loc[0, "looseness_to_bearing_rate"] == pytest.approx(1.0)
    assert scores.loc[0, "bearing_to_looseness_rate"] == pytest.approx(1.0)
    assert scores.loc[0, "looseness_other_max_offdiag_count"] == 0
    assert scores.loc[0, "bearing_other_max_offdiag_count"] == 0


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


def test_staged_record_scores_rejects_conflicting_label_within_record():
    validation = _validation()
    validation.loc[1, "label"] = LABEL_ORDER[1]

    with pytest.raises(ValueError, match=r"record.*conflict.*label"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            validation,
            ["f1", "f2"],
            [1],
            LABEL_ORDER,
        )


def test_staged_record_scores_rejects_missing_nonmissing_metadata_conflict():
    validation = _validation()
    validation.loc[1, "motor"] = None

    with pytest.raises(ValueError, match=r"record.*conflict.*motor"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            validation,
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
    assert result["fold_scores"].shape == (10, 11)
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


def _six_class_feature_train():
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
    class_number = train["label"].map({label: index for index, label in enumerate(LABEL_ORDER)}).astype(float)
    window_offset = np.tile([0.0, 0.01], len(train) // 2)
    for index, feature in enumerate(FEATURE_43):
        train[feature] = class_number + window_offset + index / 1000
    return train


def test_real_catboost_max800_emits_every_production_checkpoint():
    from catboost import CatBoostClassifier, Pool

    train = _six_class_feature_train()
    model = CatBoostClassifier(
        iterations=800,
        depth=2,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=2026,
        allow_writing_files=False,
        verbose=False,
        thread_count=1,
    )
    model.fit(Pool(train[FEATURE_40], train["label"]))

    scores = staged_record_scores(model, train, FEATURE_40, ITERATION_GRID, list(model.classes_))

    assert scores["iteration"].tolist() == ITERATION_GRID
    assert len(scores) == 40


def test_same_external_folds_are_reused_for_feature43_and_feature40(monkeypatch):
    import ablation_analysis.iteration_selection as module

    train = _six_class_feature_train()
    folds = make_record_folds(train["label"], train["record_id"])
    monkeypatch.setattr(module, "MAX_ITERATIONS", 1)

    full = select_iterations(train, FEATURE_43, folds[0], folds[1], checkpoints=[1])
    reduced = select_iterations(train, FEATURE_40, folds[0], folds[1], checkpoints=[1])

    assert full["fold_hash"] == reduced["fold_hash"] == folds[2]
    assert full["fold_scores"]["fold"].tolist() == reduced["fold_scores"]["fold"].tolist()


@pytest.mark.parametrize("column,bad_value", [("label", LABEL_ORDER[1]), ("condition", None)])
def test_select_iterations_reasserts_record_consistency_before_external_folds(
    monkeypatch, column, bad_value
):
    train = _six_class_feature_train()
    folds = make_record_folds(train["label"], train["record_id"])
    train.loc[1, column] = bad_value
    monkeypatch.setattr("ablation_analysis.iteration_selection.MAX_ITERATIONS", 1)

    with pytest.raises(ValueError, match=rf"record.*conflict.*{column}"):
        select_iterations(train, FEATURE_40, folds, checkpoints=[1])


def test_staged_record_scores_rejects_metadata_as_feature():
    validation = _validation()

    with pytest.raises(ValueError, match=r"feature_names.*rpm"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            validation,
            ["f1", "rpm"],
            [1],
            LABEL_ORDER,
        )


@pytest.mark.parametrize("bad_value", ["not-numeric", np.inf])
def test_staged_record_scores_rejects_nonnumeric_or_nonfinite_features(bad_value):
    validation = _validation()
    if isinstance(bad_value, str):
        validation["f1"] = validation["f1"].astype(object)
    validation.loc[0, "f1"] = bad_value

    with pytest.raises(ValueError, match=r"numeric.*finite"):
        staged_record_scores(
            FakeStagedModel([np.full((12, 6), 1 / 6)]),
            validation,
            ["f1", "f2"],
            [1],
            LABEL_ORDER,
        )


def _rehash_manifest(manifest):
    import ablation_analysis.iteration_selection as module

    manifest = manifest.copy()
    manifest["fold_sha256"] = module._manifest_hash(manifest)
    return manifest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.assign(extra="forbidden"),
        lambda manifest: manifest.assign(role=manifest["role"].mask(manifest.index == 0, "test")),
        lambda manifest: manifest.assign(fold=manifest["fold"] + 1),
        lambda manifest: pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True),
        lambda manifest: manifest.drop(manifest.index[0]).reset_index(drop=True),
    ],
    ids=["extra-column", "test-role", "noncanonical-folds", "duplicate", "incomplete-partition"],
)
def test_select_iterations_rejects_noncanonical_external_manifest(monkeypatch, mutate):
    train = _six_class_feature_train()
    manifest, indices, _ = make_record_folds(train["label"], train["record_id"])
    invalid = mutate(manifest.copy())
    if invalid.columns.tolist() == manifest.columns.tolist():
        invalid = _rehash_manifest(invalid)
    monkeypatch.setattr("ablation_analysis.iteration_selection.MAX_ITERATIONS", 1)

    with pytest.raises(ValueError, match="manifest"):
        select_iterations(train, FEATURE_40, invalid, indices, checkpoints=[1])


def test_select_iterations_rejects_stale_manifest_hash(monkeypatch):
    train = _six_class_feature_train()
    manifest, indices, _ = make_record_folds(train["label"], train["record_id"])
    original = manifest.loc[0, "record_id"]
    manifest.loc[manifest["record_id"].eq(original), "record_id"] = "changed"
    monkeypatch.setattr("ablation_analysis.iteration_selection.MAX_ITERATIONS", 1)

    with pytest.raises(ValueError, match="hash"):
        select_iterations(train, FEATURE_40, manifest, indices, checkpoints=[1])


@pytest.mark.parametrize(
    "replacement",
    [
        lambda values, size: values.astype(float),
        lambda values, size: values.astype(bool),
        lambda values, size: np.append(values[1:], -1),
        lambda values, size: np.append(values[1:], size),
        lambda values, size: np.append(values[1:], values[1]),
    ],
    ids=["float", "boolean", "negative", "out-of-range", "duplicate"],
)
def test_select_iterations_rejects_invalid_index_arrays(monkeypatch, replacement):
    train = _six_class_feature_train()
    manifest, indices, fold_hash = make_record_folds(train["label"], train["record_id"])
    tampered = [(fit.copy(), validation.copy()) for fit, validation in indices]
    fit, validation = tampered[0]
    tampered[0] = (replacement(fit, len(train)), validation)
    monkeypatch.setattr("ablation_analysis.iteration_selection.MAX_ITERATIONS", 1)

    with pytest.raises(ValueError, match="indices"):
        select_iterations(train, FEATURE_40, (manifest, tampered, fold_hash), checkpoints=[1])
