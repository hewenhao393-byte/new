from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pump_diagnosis.three_class_feature_table import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    extract_window_features,
    iter_windows,
    preprocess_signal,
)


DEFAULT_DATA_ROOT = Path(
    "/Users/hewenhao/Documents/电机驱动离心泵多故障电流与振动监测数据集/数据集/原始数据集/Vibration/Motor-4/70"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/hewenhao/Documents/特征提取/实验结果/Motor4_70_2070rpm_四分类_通道4_特征表.csv"
)

_LABEL_PREFIXES = {
    "正常": ("正常状态",),
    "转子不平衡": ("泵不平衡", "电机不平衡"),
    "联轴器不对中": ("角向不对中", "平行不对中", "组合不对中"),
    "汽蚀": ("吸入口汽蚀", "出口汽蚀"),
}


@dataclass(frozen=True)
class FourClassFeatureConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_path: Path = DEFAULT_OUTPUT_PATH
    original_fs: int = 20_000
    processed_fs: int = 12_000
    resample_up: int = 3
    resample_down: int = 5
    bandpass_low: float = 10.0
    bandpass_high: float = 5000.0
    filter_order: int = 4
    window_size: int = 2400
    step_size: int = 1200
    rpm: float = 2070.0
    harmonic_search_hz: float = 2.0
    wavelet: str = "db6"
    wavelet_level: int = 3
    envelope_band_low: float = 2000.0
    envelope_band_high: float = 5000.0
    csv_encoding: str = "utf-8-sig"
    summary_path: Path | None = field(default=None)

    @property
    def rotation_frequency(self) -> float:
        return self.rpm / 60.0


def map_fault_to_four_class(raw_fault_name: str) -> str | None:
    for label, prefixes in _LABEL_PREFIXES.items():
        if any(raw_fault_name.startswith(prefix) for prefix in prefixes):
            return label
    return None


def build_feature_table(config: FourClassFeatureConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(config.data_root.rglob("*通道4.csv")):
        label = map_fault_to_four_class(csv_path.parent.name)
        if label is None:
            continue
        raw = pd.read_csv(csv_path)
        signal_columns = [column for column in raw.columns if column != "time"]
        for column_name in signal_columns:
            samples = raw[column_name].to_numpy(dtype=float)
            processed = preprocess_signal(samples, config)
            for window_index, (start, end, window) in enumerate(
                iter_windows(processed, config.window_size, config.step_size)
            ):
                row = {
                    "source_file": str(csv_path),
                    "window_id": f"{csv_path.stem}_{column_name}_{window_index}",
                    "window_start": int(start),
                    "window_end": int(end),
                    "label": label,
                }
                row.update(extract_window_features(window, config))
                rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_feature_table(config: FourClassFeatureConfig) -> tuple[Path, pd.DataFrame]:
    frame = build_feature_table(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output_path, index=False, encoding=config.csv_encoding)
    summary_path = config.summary_path or config.output_path.with_suffix(".summary.json")
    summary = {
        "rows": int(len(frame)),
        "files": int(frame["source_file"].nunique()) if not frame.empty else 0,
        "labels": frame["label"].value_counts().to_dict() if not frame.empty else {},
        "window_size": config.window_size,
        "step_size": config.step_size,
        "processed_fs": config.processed_fs,
        "rpm": config.rpm,
        "rotation_frequency_hz": config.rotation_frequency,
        "wavelet": config.wavelet,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return config.output_path, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Motor-4 2070 rpm channel-4 four-class feature table.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--wavelet", choices=["db4", "db6"], default="db6")
    args = parser.parse_args()

    config = FourClassFeatureConfig(
        data_root=Path(args.data_root),
        output_path=Path(args.output_path),
        wavelet=args.wavelet,
    )
    output_path, frame = save_feature_table(config)
    print(f"saved={output_path}")
    print(f"rows={len(frame)}")
    if not frame.empty:
        print(frame["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
