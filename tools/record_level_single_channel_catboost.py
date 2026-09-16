"""Leakage-safe single-channel CatBoost training and evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


RECORD_METADATA_COLUMNS = [
    "group_id",
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]
MATCH_COLUMNS = [
    "record_column",
    "label",
    "device_id",
    "speed_percent",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]
CHANNEL_PATH_COLUMNS = ["ch3_path", "ch4_path", "ch5_path"]
METADATA_COLUMNS = [
    "group_id",
    "record_column",
    "device_id",
    "speed_percent",
    "rpm",
    "label",
    "window_id",
    "window_start",
    "window_end",
    "ch3_path",
    "ch4_path",
    "ch5_path",
]
LABEL_ORDER = ["正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"]
FIXED_PARAMS = {
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
SOURCE_ROOT = (
    Path(__file__).resolve().parents[1] / "实验结果" / "三通道融合_文档流程_5Hz"
)
OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1] / "实验结果" / "三通道单模型_CatBoost_记录级划分"
)


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {frame_name}: {missing}")


def _stable_local_seed(seed: int, source_path: object) -> int:
    digest = hashlib.sha256(f"{seed}:{source_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_record_assignment(
    records: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign complete 12-second records to train/test within each source file."""
    _require_columns(records, RECORD_METADATA_COLUMNS, "records")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    metadata = records.loc[:, RECORD_METADATA_COLUMNS].drop_duplicates().copy()
    conflicting = metadata["group_id"].duplicated(keep=False)
    if conflicting.any():
        group_ids = sorted(metadata.loc[conflicting, "group_id"].astype(str).unique())
        raise ValueError(f"group_id metadata is not unique: {group_ids}")

    unique = metadata.sort_values(["ch4_path", "group_id"], kind="stable")
    parts = []
    for source_path, group in unique.groupby("ch4_path", sort=True, dropna=False):
        group = group.sort_values("group_id", kind="stable").reset_index(drop=True)
        if len(group) < 2:
            raise ValueError(f"Source file has fewer than two records: {source_path}")
        rng = np.random.default_rng(_stable_local_seed(seed, source_path))
        shuffled = group.iloc[rng.permutation(len(group))].copy()
        test_count = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        shuffled["subset"] = "train"
        if test_count:
            shuffled.iloc[:test_count, shuffled.columns.get_loc("subset")] = "test"
        parts.append(shuffled)

    columns = RECORD_METADATA_COLUMNS + ["subset"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
        .loc[:, columns]
    )


def _equal_including_missing(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def validate_no_leakage(
    windows: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Validate record/window isolation and attach each window's subset."""
    _require_columns(windows, RECORD_METADATA_COLUMNS + ["window_id"], "windows")
    _require_columns(assignment, RECORD_METADATA_COLUMNS + ["subset"], "assignment")

    if assignment["group_id"].duplicated().any():
        raise ValueError("Duplicate group assignment causes group leakage")
    invalid_subsets = set(assignment["subset"].dropna()).difference({"train", "test"})
    if invalid_subsets or assignment["subset"].isna().any():
        raise ValueError(f"Invalid subset values: {sorted(map(str, invalid_subsets))}")
    if windows.duplicated(["group_id", "window_id"]).any():
        raise ValueError("Duplicate window key")

    path_counts = windows.groupby("group_id", dropna=False)[CHANNEL_PATH_COLUMNS].nunique(
        dropna=False
    )
    if not path_counts.eq(1).all().all():
        raise ValueError("Inconsistent channel metadata within group")

    train_ids = set(assignment.loc[assignment["subset"].eq("train"), "group_id"])
    test_ids = set(assignment.loc[assignment["subset"].eq("test"), "group_id"])
    if train_ids.intersection(test_ids):
        raise ValueError("group leakage detected between train and test")

    assignment_columns = RECORD_METADATA_COLUMNS + ["subset"]
    merged = windows.merge(
        assignment.loc[:, assignment_columns],
        on="group_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "__assignment"),
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        missing_ids = sorted(
            merged.loc[merged["_merge"].ne("both"), "group_id"].astype(str).unique()
        )
        raise ValueError(f"Some windows have no record assignment: {missing_ids}")

    for column in MATCH_COLUMNS:
        assigned_column = f"{column}__assignment"
        matches = _equal_including_missing(merged[column], merged[assigned_column])
        if not matches.all():
            group_ids = sorted(merged.loc[~matches, "group_id"].astype(str).unique())
            raise ValueError(f"Record metadata mismatch for {column}: {group_ids}")

    drop_columns = ["_merge"] + [f"{column}__assignment" for column in MATCH_COLUMNS]
    return merged.drop(columns=drop_columns)


def split_distribution(assignment: pd.DataFrame) -> pd.DataFrame:
    """Summarize record and source-file coverage for every split stratum."""
    required = [
        "subset",
        "label",
        "device_id",
        "speed_percent",
        "group_id",
        "ch4_path",
    ]
    _require_columns(assignment, required, "assignment")
    return (
        assignment.groupby(
            ["subset", "label", "device_id", "speed_percent"],
            dropna=False,
            sort=True,
        )
        .agg(records=("group_id", "nunique"), files=("ch4_path", "nunique"))
        .reset_index()
    )


def select_channel_features(names, channel: str) -> list[str]:
    """Return one channel's exact 21 features in their source-column order."""
    channel = str(channel)
    if channel not in {"3", "4", "5"}:
        raise ValueError(f"Unsupported channel: {channel}")
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names are not allowed")
    path_column = f"ch{channel}_path"
    selected = [
        name
        for name in names
        if name.startswith(f"ch{channel}_") and name != path_column
    ]
    if len(selected) != 21 or len(set(selected)) != 21:
        raise ValueError(f"Channel {channel} does not have 21 unique features")
    return selected


def fuse_record_probabilities(
    probability_frame: pd.DataFrame,
    classes,
) -> pd.DataFrame:
    """Average window probabilities by record and retain unambiguous truth."""
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must contain unique labels")
    _require_columns(probability_frame, ["group_id"] + classes, "probability_frame")
    probabilities = probability_frame.loc[:, classes].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")

    if "true_label" in probability_frame.columns:
        if probability_frame["true_label"].isna().any():
            raise ValueError("Each group must have exactly one non-missing true label")
        truth_counts = probability_frame.groupby("group_id", dropna=False)[
            "true_label"
        ].nunique(dropna=False)
        if not truth_counts.eq(1).all():
            raise ValueError("Each group must have exactly one true label")

    fused = probability_frame.groupby("group_id", sort=True, dropna=False)[classes].mean()
    if "true_label" in probability_frame.columns:
        truth = probability_frame.groupby("group_id", sort=True, dropna=False)[
            "true_label"
        ].first()
        fused["true_label"] = truth.reindex(fused.index)
    fused["predicted_label"] = fused[classes].idxmax(axis=1)
    return fused


def fit_channel_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features,
    channel: str,
):
    """Fit one frozen CatBoost configuration and return test probabilities."""
    from catboost import CatBoostClassifier, Pool

    features = list(features)
    expected_train = select_channel_features(train.columns, channel)
    expected_test = select_channel_features(test.columns, channel)
    if features != expected_train or features != expected_test:
        raise ValueError(f"Channel {channel} feature order does not match source data")
    _require_columns(train, features + ["label"], "train")
    _require_columns(test, features + ["label"], "test")
    if not np.isfinite(train[features].to_numpy(dtype=float)).all():
        raise ValueError("Training features must be finite")
    if not np.isfinite(test[features].to_numpy(dtype=float)).all():
        raise ValueError("Test features must be finite")

    train_pool = Pool(train.loc[:, features], train["label"])
    test_pool = Pool(test.loc[:, features], test["label"])
    model = CatBoostClassifier(**FIXED_PARAMS)
    model.fit(train_pool, verbose=180)
    if list(model.feature_names_) != features:
        raise RuntimeError("Fitted model feature order differs from requested feature order")
    probabilities = np.asarray(
        model.predict_proba(test_pool, thread_count=FIXED_PARAMS["thread_count"]),
        dtype=float,
    )
    if probabilities.shape != (len(test), len(model.classes_)):
        raise RuntimeError("Unexpected probability matrix shape")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model returned non-finite probabilities")
    return model, probabilities


def compute_classification_metrics(true_labels, predicted_labels, labels) -> dict:
    """Compute the complete fixed-order classification metric contract."""
    labels = list(labels)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Truth and prediction lengths differ")
    return {
        "label_order": labels,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=labels
        ).tolist(),
    }


def _validate_feature_metadata(features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    _require_columns(features, METADATA_COLUMNS, "features")
    _require_columns(metadata, METADATA_COLUMNS, "window_metadata")
    for frame_name, frame in (("features", features), ("window_metadata", metadata)):
        if frame.duplicated(["group_id", "window_id"]).any():
            raise ValueError(f"Duplicate group/window key in {frame_name}")
    if len(features) != len(metadata):
        raise ValueError("Feature and metadata window counts differ")

    left = features.loc[:, METADATA_COLUMNS].sort_values(
        ["group_id", "window_id"], kind="stable"
    ).reset_index(drop=True)
    right = metadata.loc[:, METADATA_COLUMNS].sort_values(
        ["group_id", "window_id"], kind="stable"
    ).reset_index(drop=True)
    for column in METADATA_COLUMNS:
        if not _equal_including_missing(left[column], right[column]).all():
            raise ValueError(f"Feature/metadata mismatch for {column}")


def load_complete_feature_table(source_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load all records, preferring the complete fused table over old splits."""
    source_root = Path(source_root)
    metadata_path = source_root / "window_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)

    complete_path = source_root / "channel345_fused_features.csv"
    if complete_path.exists():
        features = pd.read_csv(complete_path)
        sources = [complete_path]
    else:
        split_root = source_root / "file_group_holdout_80_20"
        train_path = split_root / "train_features_63.csv"
        test_path = split_root / "test_features_63.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Neither complete fused features nor both legacy split files exist"
            )
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        if train.columns.tolist() != test.columns.tolist():
            raise ValueError("Legacy train/test schemas differ")
        train_keys = set(zip(train["group_id"], train["window_id"]))
        test_keys = set(zip(test["group_id"], test["window_id"]))
        if train_keys.intersection(test_keys):
            raise ValueError("Legacy train/test contain overlapping windows")
        features = pd.concat([train, test], ignore_index=True)
        sources = [train_path, test_path]

    _validate_feature_metadata(features, metadata)
    return features, sources


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    """Run the fixed three-model experiment; intentionally not called on import."""
    from catboost import CatBoostClassifier, Pool

    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite output directory: {OUTPUT_ROOT}")

    data, source_paths = load_complete_feature_table(SOURCE_ROOT)
    metadata_path = SOURCE_ROOT / "window_metadata.csv"
    if len(data) != 219_555 or data["group_id"].nunique() != 1_845:
        raise ValueError(
            "Complete source must contain 219,555 windows and 1,845 records"
        )
    feature_names = [column for column in data.columns if column not in METADATA_COLUMNS]
    if len(feature_names) != 63 or len(set(feature_names)) != 63:
        raise ValueError("Complete source must contain exactly 63 unique features")
    for channel in ("3", "4", "5"):
        select_channel_features(feature_names, channel)

    records = data.loc[:, RECORD_METADATA_COLUMNS].drop_duplicates()
    assignment = build_record_assignment(records, test_fraction=0.2, seed=42)
    assigned = validate_no_leakage(data, assignment)
    if assigned["group_id"].nunique() != 1_845 or len(assigned) != 219_555:
        raise RuntimeError("Record assignment did not retain the complete dataset")
    train = assigned.loc[assigned["subset"].eq("train")].copy()
    test = assigned.loc[assigned["subset"].eq("test")].copy()
    if set(train["group_id"]).intersection(test["group_id"]):
        raise RuntimeError("Record leakage detected after assignment")

    OUTPUT_ROOT.mkdir(parents=True)
    assignment_path = OUTPUT_ROOT / "record_assignment.csv"
    assignment.to_csv(assignment_path, index=False, encoding="utf-8-sig")
    split_distribution(assignment).to_csv(
        OUTPUT_ROOT / "split_distribution.csv", index=False, encoding="utf-8-sig"
    )
    assignment_sha256 = _sha256(assignment_path)
    source_sha256 = {str(path): _sha256(path) for path in source_paths}
    source_sha256[str(metadata_path)] = _sha256(metadata_path)

    comparison_rows = []
    for channel in ("3", "4", "5"):
        channel_dir = OUTPUT_ROOT / f"ch{channel}"
        channel_dir.mkdir()
        features = select_channel_features(feature_names, channel)
        model, probabilities = fit_channel_model(
            train, test, features, channel
        )
        classes = list(model.classes_)
        model_path = channel_dir / f"ch{channel}_catboost.cbm"
        model.save_model(str(model_path))

        restored = CatBoostClassifier()
        restored.load_model(str(model_path))
        if list(restored.feature_names_) != features:
            raise RuntimeError("Reloaded model feature order differs")
        restored_probabilities = restored.predict_proba(
            Pool(test.loc[:, features], test["label"]),
            thread_count=FIXED_PARAMS["thread_count"],
        )
        np.testing.assert_allclose(
            probabilities, restored_probabilities, rtol=0, atol=1e-12
        )

        window_predictions = test.loc[
            :, ["group_id", "window_id", "label"]
        ].reset_index(drop=True)
        window_predictions = window_predictions.rename(columns={"label": "true_label"})
        window_predictions = pd.concat(
            [
                window_predictions,
                pd.DataFrame(probabilities, columns=classes),
            ],
            axis=1,
        )
        window_predictions["predicted_label"] = np.asarray(classes)[
            probabilities.argmax(axis=1)
        ]
        record_predictions = fuse_record_probabilities(window_predictions, classes)

        window_metrics = compute_classification_metrics(
            window_predictions["true_label"],
            window_predictions["predicted_label"],
            LABEL_ORDER,
        )
        record_metrics = compute_classification_metrics(
            record_predictions["true_label"],
            record_predictions["predicted_label"],
            LABEL_ORDER,
        )

        window_predictions.to_csv(
            channel_dir / "window_predictions.csv", index=False, encoding="utf-8-sig"
        )
        record_predictions.to_csv(
            channel_dir / "record_predictions.csv", encoding="utf-8-sig"
        )
        _write_json(channel_dir / "window_metrics.json", window_metrics)
        _write_json(channel_dir / "record_metrics.json", record_metrics)
        _write_json(
            channel_dir / "metadata.json",
            {
                "channel": channel,
                "features": features,
                "classes": classes,
                "label_order": LABEL_ORDER,
                "fixed_params": FIXED_PARAMS,
                "source_sha256": source_sha256,
                "assignment_sha256": assignment_sha256,
                "train_windows": len(train),
                "test_windows": len(test),
                "train_records": train["group_id"].nunique(),
                "test_records": test["group_id"].nunique(),
                "model_reload_verified": True,
            },
        )
        comparison_rows.append(
            {
                "channel": channel,
                "feature_count": len(features),
                "window_accuracy": window_metrics["accuracy"],
                "window_balanced_accuracy": window_metrics["balanced_accuracy"],
                "window_macro_f1": window_metrics["macro_f1"],
                "record_accuracy": record_metrics["accuracy"],
                "record_balanced_accuracy": record_metrics["balanced_accuracy"],
                "record_macro_f1": record_metrics["macro_f1"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_ROOT / "comparison.csv", index=False, encoding="utf-8-sig")
    _write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "purpose": "same-file record-level evaluation",
            "channels": ["3", "4", "5"],
            "test_fraction": 0.2,
            "seed": 42,
            "no_tuning": True,
            "early_stopping": False,
            "source_sha256": source_sha256,
            "assignment_sha256": assignment_sha256,
            "results": comparison_rows,
        },
    )


if __name__ == "__main__":
    main()
