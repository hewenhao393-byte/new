from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig
from pump_diagnosis.features import (
    ENVELOPE_FEATURES,
    FEATURE_COLUMNS,
    FREQUENCY_FEATURES,
    ROTATION_FEATURES,
    TIME_FEATURES,
    WAVELET_FEATURES,
    extract_candidate_features,
)
from pump_diagnosis.signal_processing import iter_windows, preprocess_run


METADATA_COLUMNS = [
    "sample_id",
    "label",
    "run_id",
    "file_path",
    "machine_id",
    "condition_id",
    "rpm",
    "channel",
    "window_index",
    "window_start",
    "window_end",
    "original_fs",
    "processed_fs",
]


@dataclass(frozen=True)
class Stage1Result:
    feature_csv: Path
    quality_csv: Path
    summary_json: Path


def extract_split_features(
    manifest: pd.DataFrame,
    split_name: str,
    config: PipelineConfig,
) -> Stage1Result:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    feature_csv = output_root / f"{split_name}_features_raw.csv"
    quality_csv = output_root / f"{split_name}_quality_report.csv"
    summary_json = output_root / f"{split_name}_feature_summary.json"
    result = Stage1Result(feature_csv=feature_csv, quality_csv=quality_csv, summary_json=summary_json)
    if feature_csv.exists() and quality_csv.exists() and summary_json.exists():
        return result

    feature_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for file_path, file_rows in manifest.groupby("file_path", sort=True):
        raw = pd.read_csv(file_path)
        for row in file_rows.itertuples(index=False):
            samples = raw[str(row.source_column)].to_numpy(dtype=np.float64)
            if not np.isfinite(samples).all():
                quality_rows.append({"run_id": row.run_id, "reason": "raw_nonfinite"})
                continue
            processed = preprocess_run(samples, config)
            for window_index, (start, end, window) in enumerate(
                iter_windows(processed, config.window_size, config.step_size)
            ):
                feature_row = {
                    "sample_id": f"{row.run_id}_window_{window_index}",
                    "label": row.label,
                    "run_id": row.run_id,
                    "file_path": row.file_path,
                    "machine_id": row.machine_id,
                    "condition_id": row.condition_id,
                    "rpm": row.rpm,
                    "channel": row.channel,
                    "window_index": window_index,
                    "window_start": start / config.processed_fs,
                    "window_end": end / config.processed_fs,
                    "original_fs": row.original_fs,
                    "processed_fs": config.processed_fs,
                }
                feature_row.update(extract_candidate_features(window, rpm=row.rpm, config=config))
                feature_rows.append(feature_row)

    pd.DataFrame(feature_rows, columns=METADATA_COLUMNS + FEATURE_COLUMNS).to_csv(
        feature_csv,
        index=False,
    )
    pd.DataFrame(quality_rows).to_csv(quality_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "rows": len(feature_rows),
                "skipped_runs": len(quality_rows),
                "feature_count": len(FEATURE_COLUMNS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def write_feature_dictionary(path: Path) -> None:
    rows: list[dict[str, str]] = []
    rows.extend(_dictionary_rows(TIME_FEATURES, "time", "时域"))
    rows.extend(_dictionary_rows(FREQUENCY_FEATURES, "frequency", "频域"))
    rows.extend(_dictionary_rows(ROTATION_FEATURES, "rotation", "转频"))
    rows.extend(_dictionary_rows(WAVELET_FEATURES, "wavelet", "小波包"))
    rows.extend(_dictionary_rows(ENVELOPE_FEATURES, "envelope", "包络"))
    pd.DataFrame(rows).to_csv(path, index=False)


def _dictionary_rows(feature_names: list[str], group_key: str, group_name: str) -> list[dict[str, str]]:
    return [
        {
            "feature_name": feature_name,
            "chinese_name": f"{group_name}_{feature_name}",
            "feature_group": group_key,
        }
        for feature_name in feature_names
    ]
