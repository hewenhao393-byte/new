import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.record_level_single_channel_catboost import (
    FIXED_PARAMS,
    build_record_assignment,
    compute_classification_metrics,
    fit_channel_model,
    fuse_record_probabilities,
    load_complete_feature_table,
    run_experiment,
    select_channel_features,
    split_distribution,
    validate_no_leakage,
)


RECORD_COLUMNS = [
    "group_id",
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]


@pytest.fixture
def example_records():
    rows = []
    specifications = [
        ("source/a_ch4.csv", 5, "正常", "Motor-2", 100),
        ("source/b_ch4.csv", 2, "汽蚀", "Motor-4", 70),
    ]
    for source_index, (ch4_path, count, label, device_id, speed) in enumerate(
        specifications
    ):
        for record_index in range(count):
            rows.append(
                {
                    "group_id": f"g{source_index}-{record_index}",
                    "record_column": str(record_index),
                    "label": label,
                    "device_id": device_id,
                    "speed_percent": speed,
                    "ch3_path": ch4_path.replace("ch4", "ch3"),
                    "ch4_path": ch4_path,
                    "ch5_path": ch4_path.replace("ch4", "ch5"),
                }
            )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


@pytest.fixture
def example_assignment(example_records):
    return build_record_assignment(example_records, test_fraction=0.2, seed=42)


@pytest.fixture
def example_windows(example_records):
    rows = []
    for record in example_records.to_dict("records"):
        for window_id in range(2):
            rows.append({**record, "window_id": window_id, "feature": window_id + 0.5})
    return pd.DataFrame(rows)


def test_split_keeps_every_record_and_is_deterministic(example_records):
    first = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    shuffled = example_records.sample(frac=1, random_state=7).reset_index(drop=True)
    second = build_record_assignment(shuffled, test_fraction=0.2, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["group_id"].is_unique
    assert set(first["group_id"]) == set(example_records["group_id"])


def test_each_eligible_file_contributes_train_and_test_records(example_records):
    assignment = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    counts = (
        assignment.groupby(["ch4_path", "subset"])["group_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    assert (counts["train"] > 0).all()
    assert (counts["test"] > 0).all()
    assert counts.loc["source/a_ch4.csv", "test"] == 1
    assert counts.loc["source/b_ch4.csv", "test"] == 1


def test_split_uses_requested_rounded_test_count(example_records):
    assignment = build_record_assignment(example_records, test_fraction=0.6, seed=42)
    counts = assignment.groupby(["ch4_path", "subset"])["group_id"].nunique()

    assert counts.loc[("source/a_ch4.csv", "test")] == 3
    assert counts.loc[("source/a_ch4.csv", "train")] == 2
    assert counts.loc[("source/b_ch4.csv", "test")] == 1
    assert counts.loc[("source/b_ch4.csv", "train")] == 1


def test_split_rejects_missing_required_columns(example_records):
    with pytest.raises(ValueError, match="Missing columns.*ch5_path"):
        build_record_assignment(example_records.drop(columns="ch5_path"))


def test_split_rejects_source_file_with_fewer_than_two_records(example_records):
    single_record = example_records.iloc[[0]].copy()

    with pytest.raises(ValueError, match="fewer than two records"):
        build_record_assignment(single_record)


@pytest.mark.parametrize("test_fraction", [-0.1, 0, 1, 1.1])
def test_split_rejects_test_fraction_outside_open_interval(
    example_records, test_fraction
):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build_record_assignment(example_records, test_fraction=test_fraction)


def test_split_rejects_conflicting_group_metadata(example_records):
    conflicting = pd.concat(
        [example_records, example_records.iloc[[0]].assign(label="轴承故障")],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="group_id metadata"):
        build_record_assignment(conflicting)


def test_validate_returns_all_windows_with_subset(
    example_windows, example_assignment
):
    merged = validate_no_leakage(example_windows, example_assignment)

    assert len(merged) == len(example_windows)
    assert set(merged["subset"]) == {"train", "test"}
    assert merged[["group_id", "window_id"]].equals(
        example_windows[["group_id", "window_id"]]
    )


def test_validate_rejects_duplicate_group_assignment(example_windows, example_assignment):
    broken = pd.concat([example_assignment, example_assignment.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate group assignment"):
        validate_no_leakage(example_windows, broken)


def test_validate_rejects_missing_assignment(example_windows, example_assignment):
    broken = example_assignment.iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="no record assignment"):
        validate_no_leakage(example_windows, broken)


def test_validate_rejects_duplicate_window_key(example_windows, example_assignment):
    broken = pd.concat([example_windows, example_windows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate window key"):
        validate_no_leakage(broken, example_assignment)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("record_column", "wrong-record"),
        ("label", "轴承故障"),
        ("device_id", "wrong-device"),
        ("speed_percent", 999),
        ("ch3_path", "/wrong/channel3.csv"),
        ("ch4_path", "/wrong/channel4.csv"),
        ("ch5_path", "/wrong/channel5.csv"),
    ],
)
def test_validate_rejects_path_or_metadata_mismatch(
    example_windows, example_assignment, column, replacement
):
    broken = example_windows.copy()
    group_id = broken.loc[0, "group_id"]
    broken.loc[broken["group_id"].eq(group_id), column] = replacement

    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_no_leakage(broken, example_assignment)


def test_validate_rejects_unstable_channel_path_within_group(
    example_windows, example_assignment
):
    broken = example_windows.copy()
    broken.loc[0, "ch3_path"] = "/wrong/channel3.csv"

    with pytest.raises(ValueError, match="Inconsistent channel metadata"):
        validate_no_leakage(broken, example_assignment)


def test_validate_rejects_train_test_group_overlap(example_windows, example_assignment):
    overlapping = pd.concat(
        [
            example_assignment,
            example_assignment.iloc[[0]].assign(subset=lambda frame: frame["subset"].map(
                {"train": "test", "test": "train"}
            )),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="group leakage|Duplicate group assignment"):
        validate_no_leakage(example_windows, overlapping)


def test_split_distribution_counts_records_and_files(example_assignment):
    distribution = split_distribution(example_assignment)
    expected_columns = [
        "subset",
        "label",
        "device_id",
        "speed_percent",
        "records",
        "files",
    ]

    assert distribution.columns.tolist() == expected_columns
    assert distribution["records"].sum() == len(example_assignment)
    assert (distribution["files"] == 1).all()


@pytest.fixture
def formal_feature_names():
    base = [f"feature_{index:02d}" for index in range(21)]
    return [f"ch{channel}_{name}" for channel in (3, 4, 5) for name in base]


@pytest.mark.parametrize("channel", ["3", "4", "5"])
def test_select_channel_features_returns_exact_source_order(
    formal_feature_names, channel
):
    selected = select_channel_features(formal_feature_names, channel)

    assert len(selected) == 21
    assert selected == [
        name for name in formal_feature_names if name.startswith(f"ch{channel}_")
    ]


def test_select_channel_features_rejects_unsupported_channel(formal_feature_names):
    with pytest.raises(ValueError, match="Unsupported channel"):
        select_channel_features(formal_feature_names, "2")


def test_select_channel_features_rejects_incomplete_or_duplicate_names(
    formal_feature_names,
):
    incomplete = formal_feature_names[:-1]
    duplicate = formal_feature_names + [formal_feature_names[0]]

    with pytest.raises(ValueError, match="21 unique"):
        select_channel_features(incomplete, "5")
    with pytest.raises(ValueError, match="Duplicate feature names"):
        select_channel_features(duplicate, "3")


def test_select_channel_features_ignores_channel_path_metadata(formal_feature_names):
    names = ["group_id", "ch3_path", *formal_feature_names, "ch4_path", "ch5_path"]

    selected = select_channel_features(names, "3")

    assert len(selected) == 21
    assert "ch3_path" not in selected


def test_fuse_record_probabilities_averages_windows_and_preserves_truth():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "b"],
            "true_label": ["正常", "正常", "汽蚀"],
            "正常": [0.8, 0.6, 0.1],
            "汽蚀": [0.2, 0.4, 0.9],
        }
    )

    fused = fuse_record_probabilities(frame, ["正常", "汽蚀"])

    assert fused.index.tolist() == ["a", "b"]
    assert fused.loc["a", "正常"] == pytest.approx(0.7)
    assert fused.loc["a", "predicted_label"] == "正常"
    assert fused.loc["b", "true_label"] == "汽蚀"


def test_fuse_record_probabilities_rejects_invalid_input():
    missing_class = pd.DataFrame({"group_id": ["a"], "正常": [1.0]})
    non_finite = pd.DataFrame(
        {"group_id": ["a"], "正常": [np.nan], "汽蚀": [0.0]}
    )
    inconsistent_truth = pd.DataFrame(
        {
            "group_id": ["a", "a"],
            "true_label": ["正常", "汽蚀"],
            "正常": [0.8, 0.7],
            "汽蚀": [0.2, 0.3],
        }
    )
    missing_truth = pd.DataFrame(
        {
            "group_id": ["a"],
            "true_label": [None],
            "正常": [0.8],
            "汽蚀": [0.2],
        }
    )

    with pytest.raises(ValueError, match="Missing columns"):
        fuse_record_probabilities(missing_class, ["正常", "汽蚀"])
    with pytest.raises(ValueError, match="finite"):
        fuse_record_probabilities(non_finite, ["正常", "汽蚀"])
    with pytest.raises(ValueError, match="true label"):
        fuse_record_probabilities(inconsistent_truth, ["正常", "汽蚀"])
    with pytest.raises(ValueError, match="true label"):
        fuse_record_probabilities(missing_truth, ["正常", "汽蚀"])


def test_fixed_params_match_frozen_experiment_contract():
    assert FIXED_PARAMS == {
        "loss_function": "MultiClass",
        "iterations": 540,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 100,
        "random_strength": 5,
        "rsm": 0.7,
        "auto_class_weights": "SqrtBalanced",
        "random_seed": 42,
        "thread_count": 4,
        "allow_writing_files": False,
    }


def test_fit_channel_model_uses_exact_feature_order(monkeypatch, formal_feature_names):
    features = select_channel_features(formal_feature_names, "3")
    train = pd.DataFrame(
        [[0.0] * 21, [1.0] * 21], columns=features
    ).assign(label=["正常", "汽蚀"])
    test = pd.DataFrame([[0.5] * 21], columns=features).assign(label=["正常"])
    calls = {}

    class FakePool:
        def __init__(self, data, label=None):
            calls.setdefault("pool_columns", []).append(data.columns.tolist())
            self.data = data
            self.label = label

    class FakeModel:
        def __init__(self, **params):
            calls["params"] = params
            self.feature_names_ = []
            self.classes_ = np.array(["正常", "汽蚀"])

        def fit(self, pool, verbose=None):
            calls["fit_verbose"] = verbose
            self.feature_names_ = pool.data.columns.tolist()
            return self

        def predict_proba(self, pool, thread_count=None):
            calls["predict_thread_count"] = thread_count
            return np.tile([0.75, 0.25], (len(pool.data), 1))

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        types.SimpleNamespace(CatBoostClassifier=FakeModel, Pool=FakePool),
    )

    model, probabilities = fit_channel_model(train, test, features, "3")

    assert model.feature_names_ == features
    assert calls["pool_columns"] == [features, features]
    assert calls["params"] == FIXED_PARAMS
    assert probabilities.shape == (1, 2)


def test_fit_channel_model_rejects_wrong_channel_feature_order(formal_feature_names):
    features = select_channel_features(formal_feature_names, "3")
    train = pd.DataFrame([[0.0] * 21], columns=features).assign(label=["正常"])
    test = train.copy()

    with pytest.raises(ValueError, match="feature order"):
        fit_channel_model(train, test, list(reversed(features)), "3")


def test_compute_classification_metrics_has_full_contract():
    labels = ["正常", "汽蚀"]
    metrics = compute_classification_metrics(
        ["正常", "正常", "汽蚀"],
        ["正常", "汽蚀", "汽蚀"],
        labels,
    )

    assert metrics["label_order"] == labels
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert set(labels).issubset(metrics["classification_report"])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 1]]


def test_load_complete_feature_table_prefers_and_validates_full_table(
    tmp_path, formal_feature_names
):
    metadata = pd.DataFrame(
        {
            "group_id": ["a", "b"],
            "record_column": ["0", "1"],
            "device_id": ["Motor-2", "Motor-4"],
            "speed_percent": [100, 70],
            "rpm": [1480, 2070],
            "label": ["正常", "汽蚀"],
            "window_id": [0, 0],
            "window_start": [0, 0],
            "window_end": [2400, 2400],
            "ch3_path": ["a3", "b3"],
            "ch4_path": ["a4", "b4"],
            "ch5_path": ["a5", "b5"],
        }
    )
    features = metadata.copy()
    for index, name in enumerate(formal_feature_names):
        features[name] = float(index)
    metadata.to_csv(tmp_path / "window_metadata.csv", index=False)
    features.to_csv(tmp_path / "channel345_fused_features.csv", index=False)

    loaded, sources = load_complete_feature_table(tmp_path)

    assert loaded.shape == features.shape
    assert loaded["group_id"].tolist() == ["a", "b"]
    assert sources == [tmp_path / "channel345_fused_features.csv"]


def test_load_complete_feature_table_combines_disjoint_legacy_splits(tmp_path):
    split = tmp_path / "file_group_holdout_80_20"
    split.mkdir()
    metadata = pd.DataFrame(
        {
            "group_id": ["a", "b"],
            "record_column": ["0", "1"],
            "device_id": ["Motor-2", "Motor-4"],
            "speed_percent": [100, 70],
            "rpm": [1480, 2070],
            "label": ["正常", "汽蚀"],
            "window_id": [0, 0],
            "window_start": [0, 0],
            "window_end": [2400, 2400],
            "ch3_path": ["a3", "b3"],
            "ch4_path": ["a4", "b4"],
            "ch5_path": ["a5", "b5"],
            "ch3_feature": [0.1, 0.2],
        }
    )
    metadata.iloc[[0]].to_csv(split / "train_features_63.csv", index=False)
    metadata.iloc[[1]].to_csv(split / "test_features_63.csv", index=False)
    metadata.drop(columns="ch3_feature").to_csv(
        tmp_path / "window_metadata.csv", index=False
    )

    loaded, sources = load_complete_feature_table(tmp_path)

    assert loaded["group_id"].tolist() == ["a", "b"]
    assert sources == [
        split / "train_features_63.csv",
        split / "test_features_63.csv",
    ]


@pytest.fixture
def tiny_experiment_data(formal_feature_names):
    rows = []
    sources = [
        ("source/normal_ch4.csv", "正常", "Motor-2", 100, 1480),
        ("source/cavitation_ch4.csv", "汽蚀", "Motor-4", 70, 2070),
    ]
    for source_index, (ch4_path, label, device, speed, rpm) in enumerate(sources):
        for record_index in range(2):
            group_id = f"g{source_index}-{record_index}"
            for window_id in range(2):
                row = {
                    "group_id": group_id,
                    "record_column": str(record_index),
                    "device_id": device,
                    "speed_percent": speed,
                    "rpm": rpm,
                    "label": label,
                    "window_id": window_id,
                    "window_start": window_id * 1200,
                    "window_end": window_id * 1200 + 2400,
                    "ch3_path": ch4_path.replace("ch4", "ch3"),
                    "ch4_path": ch4_path,
                    "ch5_path": ch4_path.replace("ch4", "ch5"),
                }
                row.update(
                    {
                        name: float(source_index + record_index + window_id + offset)
                        for offset, name in enumerate(formal_feature_names)
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_run_experiment_refuses_existing_output_directory(
    tmp_path, tiny_experiment_data
):
    output = tmp_path / "existing"
    output.mkdir()
    source = tmp_path / "source.csv"
    metadata = tmp_path / "window_metadata.csv"
    source.write_text("source", encoding="utf-8")
    metadata.write_text("metadata", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_experiment(
            tiny_experiment_data,
            [source],
            metadata,
            output,
        )


def test_run_experiment_reuses_assignment_and_writes_complete_outputs(
    tmp_path, tiny_experiment_data
):
    output = tmp_path / "result"
    source = tmp_path / "source.csv"
    metadata_path = tmp_path / "window_metadata.csv"
    source.write_text("source-data", encoding="utf-8")
    metadata_path.write_text("metadata-data", encoding="utf-8")
    fit_calls = []
    reload_calls = []

    class FakeModel:
        def __init__(self, channel, features):
            self.channel = channel
            self.feature_names_ = list(features)
            self.classes_ = np.array(["正常", "汽蚀"])

        def save_model(self, path):
            Path(path).write_text(f"model-{self.channel}", encoding="utf-8")

    def fake_fit(train, test, features, channel):
        fit_calls.append(
            {
                "channel": channel,
                "features": list(features),
                "train_object": id(train),
                "test_object": id(test),
                "train_groups": frozenset(train["group_id"]),
                "test_groups": frozenset(test["group_id"]),
            }
        )
        probabilities = np.tile([0.75, 0.25], (len(test), 1))
        return FakeModel(channel, features), probabilities

    def fake_reload(model_path, test, features):
        channel = model_path.parent.name.removeprefix("ch")
        reload_calls.append(
            {
                "channel": channel,
                "features": list(features),
                "test_groups": frozenset(test["group_id"]),
            }
        )
        return list(features), np.tile([0.75, 0.25], (len(test), 1))

    run_experiment(
        tiny_experiment_data,
        [source],
        metadata_path,
        output,
        fit_model=fake_fit,
        reload_predict=fake_reload,
    )

    assert [call["channel"] for call in fit_calls] == ["3", "4", "5"]
    assert [call["channel"] for call in reload_calls] == ["3", "4", "5"]
    assert len({call["train_object"] for call in fit_calls}) == 1
    assert len({call["test_object"] for call in fit_calls}) == 1
    assert len({call["train_groups"] for call in fit_calls}) == 1
    assert len({call["test_groups"] for call in fit_calls}) == 1
    assert all(len(call["features"]) == 21 for call in fit_calls)
    assert all(
        all(name.startswith(f"ch{call['channel']}_") for name in call["features"])
        for call in fit_calls
    )
    assert [call["features"] for call in reload_calls] == [
        call["features"] for call in fit_calls
    ]

    top_level = {
        "record_assignment.csv",
        "split_distribution.csv",
        "comparison.csv",
        "summary.json",
    }
    assert top_level.issubset({path.name for path in output.iterdir()})
    per_channel = {
        "metadata.json",
        "window_predictions.csv",
        "record_predictions.csv",
        "window_metrics.json",
        "record_metrics.json",
    }
    record_group_ids = []
    for channel in ("3", "4", "5"):
        channel_dir = output / f"ch{channel}"
        assert (channel_dir / f"ch{channel}_catboost.cbm").exists()
        assert per_channel.issubset({path.name for path in channel_dir.iterdir()})
        channel_metadata = json.loads((channel_dir / "metadata.json").read_text())
        assert channel_metadata["source_sha256"]
        assert len(channel_metadata["assignment_sha256"]) == 64
        assert channel_metadata["model_reload_verified"] is True
        record_group_ids.append(
            set(pd.read_csv(channel_dir / "record_predictions.csv")["group_id"])
        )
    assert record_group_ids[0] == record_group_ids[1] == record_group_ids[2]
