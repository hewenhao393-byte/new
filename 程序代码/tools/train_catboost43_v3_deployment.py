from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROGRAM_ROOT = Path(__file__).resolve().parents[1]
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "实验结果" / "单通道43维_两种划分_20260920"
DEPLOYMENT_SOURCES = {
    "CH3": SOURCE_ROOT / "file_split/features_ch3.csv",
    "CH4": SOURCE_ROOT / "file_split/features_ch4.csv",
    "CH5": SOURCE_ROOT / "file_split/features_ch5.csv",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "catboost43_v3"
KEY_COLUMNS = ("record_id", "window_id")
REQUIRED_METADATA = ("record_id", "group_id", "window_id", "channel", "label")
FIXED_PARAMS: dict[str, Any] = {
    "loss_function": "MultiClass",
    "iterations": 500,
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 100,
    "random_strength": 5,
    "rsm": 0.7,
    "auto_class_weights": "SqrtBalanced",
    "random_seed": 2026,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_tables(source_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        channel: source_root / "file_split" / f"features_{channel.lower()}.csv"
        for channel in ("CH3", "CH4", "CH5")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"deployment feature tables are missing: {missing}")
    tables = {channel: pd.read_csv(path, low_memory=False) for channel, path in paths.items()}
    validate_feature_tables(tables)
    return tables, paths


def validate_feature_tables(tables: dict[str, pd.DataFrame]) -> None:
    if tuple(tables) != ("CH3", "CH4", "CH5"):
        raise ValueError("feature tables must contain CH3, CH4, CH5 in order")
    reference_keys: list[tuple[object, object]] | None = None
    for channel, frame in tables.items():
        missing = sorted(set(REQUIRED_METADATA + FORMAL_FEATURE_NAMES).difference(frame.columns))
        if missing:
            raise ValueError(f"{channel} table is missing columns: {missing}")
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"{channel} contains duplicate window keys")
        expected_channel = int(channel[-1])
        if set(frame["channel"].astype(int)) != {expected_channel}:
            raise ValueError(f"{channel} contains mismatched channel metadata")
        if set(frame["label"].astype(str)) != set(FORMAL_LABEL_ORDER):
            raise ValueError(f"{channel} does not contain the exact six-label set")
        feature_values = frame.loc[:, FORMAL_FEATURE_NAMES].to_numpy(dtype=np.float64)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"{channel} contains non-finite features")
        keys = list(frame.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("sibling channel keys are not aligned")


def train_channel(frame: pd.DataFrame):
    from catboost import CatBoostClassifier, Pool

    features = frame.loc[:, FORMAL_FEATURE_NAMES]
    labels = frame["label"].astype(str)
    model = CatBoostClassifier(**FIXED_PARAMS)
    model.fit(Pool(features, label=labels, feature_names=list(FORMAL_FEATURE_NAMES)))
    if tuple(model.feature_names_) != FORMAL_FEATURE_NAMES:
        raise RuntimeError("trained model feature order mismatch")
    return model


def validate_saved_model(model_path: Path, frame: pd.DataFrame) -> tuple[str, ...]:
    from catboost import CatBoostClassifier

    restored = CatBoostClassifier()
    restored.load_model(str(model_path))
    if tuple(restored.feature_names_) != FORMAL_FEATURE_NAMES:
        raise RuntimeError(f"reloaded model feature order mismatch: {model_path.name}")
    classes = tuple(str(label) for label in restored.classes_)
    if len(classes) != 6 or set(classes) != set(FORMAL_LABEL_ORDER):
        raise RuntimeError(f"reloaded model class set mismatch: {model_path.name}")
    probabilities = np.asarray(
        restored.predict_proba(frame.loc[:, FORMAL_FEATURE_NAMES].head(32)),
        dtype=np.float64,
    )
    if probabilities.shape[1] != 6 or not np.isfinite(probabilities).all():
        raise RuntimeError(f"invalid probability matrix: {model_path.name}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"probabilities do not sum to 1: {model_path.name}")
    return classes


def build_manifest(
    tables: dict[str, pd.DataFrame],
    source_paths: dict[str, Path],
    model_paths: dict[str, Path],
    model_classes: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    return {
        "software_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "contract_version": INFERENCE_CONTRACT_VERSION,
        "training_scope": "all_accepted_data",
        "intended_use": "software_inference_only",
        "evaluation_claims": None,
        "parameter_statement": "P1@500 selected within the predefined candidate range; not a global optimum claim",
        "features": list(FORMAL_FEATURE_NAMES),
        "labels": list(FORMAL_LABEL_ORDER),
        "params": FIXED_PARAMS,
        "sources": {
            channel: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for channel, path in source_paths.items()
        },
        "models": {
            channel: {
                "file": path.name,
                "sha256": sha256_file(path),
                "classes": list(model_classes[channel]),
                "rows": int(len(tables[channel])),
                "records": int(tables[channel]["record_id"].nunique()),
                "class_distribution": dict(Counter(tables[channel]["label"].astype(str))),
            }
            for channel, path in model_paths.items()
        },
        "parent_group_evaluation_reused": False,
    }


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def train_and_publish(source_root: Path, output_dir: Path) -> Path:
    tables, source_paths = load_feature_tables(source_root)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing deployment directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        model_paths: dict[str, Path] = {}
        model_classes: dict[str, tuple[str, ...]] = {}
        for channel in ("CH3", "CH4", "CH5"):
            model = train_channel(tables[channel])
            model_path = temporary_dir / f"{channel.lower()}.cbm"
            model.save_model(str(model_path))
            model_paths[channel] = model_path
            model_classes[channel] = validate_saved_model(model_path, tables[channel])
        manifest = build_manifest(tables, source_paths, model_paths, model_classes)
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in (*model_paths.values(), manifest_path):
            fsync_file(path)
        fsync_directory(temporary_dir)
        os.replace(temporary_dir, output_dir)
        fsync_directory(output_dir.parent)
    except Exception:
        for path in temporary_dir.glob("*"):
            path.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    published = train_and_publish(args.source_root, args.output_dir)
    print(json.dumps({"status": "published", "output_dir": str(published)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
