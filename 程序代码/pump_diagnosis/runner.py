from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FINAL_RESULTS_ROOT, ModelingConfig, PipelineConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.metadata import ConditionCatalog, build_file_index, split_file_index
from pump_diagnosis.pipeline import METADATA_COLUMNS, extract_split_features


def _load_condition_catalog(config: PipelineConfig) -> ConditionCatalog | None:
    if not config.condition_workbook.exists():
        return None
    return ConditionCatalog.from_workbook(config.condition_workbook)


def validate_feature_csv(path, config: PipelineConfig) -> dict[str, object]:
    frame = pd.read_csv(path)
    expected_columns = METADATA_COLUMNS + FEATURE_COLUMNS
    if list(frame.columns) != expected_columns:
        raise ValueError("特征表字段顺序不正确")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("特征表存在NaN或Inf")
    wavelet_sum = frame[[f"wp_energy_{name}" for name in ("aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd")]].sum(axis=1)
    if not np.allclose(wavelet_sum.to_numpy(), 1.0, atol=config.wavelet_ratio_tolerance):
        raise ValueError("小波包能量占比之和不为1")
    return {
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "labels": frame["label"].value_counts().to_dict(),
    }


def run_inspect_stage(config: PipelineConfig) -> dict[str, object]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.data_root.exists():
        catalog = _load_condition_catalog(config)
        index = build_file_index(config, catalog=catalog)
        summary = {
            "rows": int(len(index)),
            "files": int(index["file_path"].nunique()) if not index.empty else 0,
            "labels": index["label"].value_counts().to_dict() if not index.empty else {},
        }
    else:
        summary = {"rows": 0, "files": 0, "labels": {}}
    (config.output_root / "inspect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_split_stage(config: PipelineConfig) -> dict[str, object]:
    catalog = _load_condition_catalog(config)
    index = build_file_index(config, catalog=catalog)
    train, test, report = split_file_index(index, config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.output_root / "train_manifest.csv", index=False)
    test.to_csv(config.output_root / "test_manifest.csv", index=False)
    (config.output_root / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def run_features_stage(config: PipelineConfig) -> dict[str, object]:
    train_manifest = pd.read_csv(config.output_root / "train_manifest.csv")
    test_manifest = pd.read_csv(config.output_root / "test_manifest.csv")
    train_result = extract_split_features(train_manifest, "train", config)
    test_result = extract_split_features(test_manifest, "test", config)
    return {
        "train_features": str(train_result.feature_csv),
        "test_features": str(test_result.feature_csv),
    }


def run_train_stage(config: PipelineConfig) -> dict[str, object]:
    from pump_diagnosis.modeling_runner import run_full_modeling

    modeling_config = ModelingConfig(feature_root=config.output_root, output_root=config.output_root / "modeling")
    return run_full_modeling(modeling_config)


def run_channel34_stage(config: PipelineConfig, feature_root: Path) -> dict[str, object]:
    from pump_diagnosis.channel34_fusion import run_channel34_fusion_pipeline

    modeling_config = ModelingConfig(feature_root=config.output_root, output_root=config.output_root / "modeling")
    return run_channel34_fusion_pipeline(feature_root, config, modeling_config)


def run_single_stage(stage: str, config: PipelineConfig) -> dict[str, object]:
    if stage == "inspect":
        return run_inspect_stage(config)
    if stage == "split":
        return run_split_stage(config)
    if stage in {"preprocess", "features"}:
        return run_features_stage(config)
    if stage in {"select", "train", "evaluate"}:
        return run_train_stage(config)
    if stage == "channel34":
        return run_channel34_stage(config, FINAL_RESULTS_ROOT)
    raise ValueError(f"未知阶段: {stage}")


def run_all_stages(config: PipelineConfig) -> dict[str, object]:
    run_inspect_stage(config)
    run_split_stage(config)
    run_features_stage(config)
    return run_train_stage(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Channel-4 six-class pump diagnosis pipeline.")
    parser.add_argument(
        "--stage",
        choices=["inspect", "split", "preprocess", "features", "select", "train", "evaluate", "channel34", "all"],
        required=True,
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--feature-root", default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.output_root:
        kwargs["output_root"] = Path(args.output_root)
    if args.data_root:
        kwargs["data_root"] = Path(args.data_root)
    config = PipelineConfig(**kwargs) if kwargs else PipelineConfig()
    if args.stage == "all":
        run_all_stages(config)
    elif args.stage == "channel34":
        feature_root = Path(args.feature_root) if args.feature_root else FINAL_RESULTS_ROOT
        run_channel34_stage(config, feature_root)
    else:
        run_single_stage(args.stage, config)


if __name__ == "__main__":
    main()
