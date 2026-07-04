from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import PipelineConfig
from pump_diagnosis.labels import map_fault_to_label


_WORKBOOK_COLUMN_RENAMES = {
    "Setup": "setup",
    "Failure description": "failure_description",
    "Speed (%)": "speed_percent",
    "Speed (RPM)": "speed_rpm",
    "Speed (RPM) +- 5": "speed_rpm",
}

_WORKBOOK_FAILURE_MAP = {
    "Healthy": "正常状态",
    "Healthy noise": "正常加噪声",
    "Healthy 1": "正常状态",
    "Healthy 2": "正常状态",
    "Healthy 3": "正常状态",
    "Cavitation suction": "吸入口汽蚀",
    "Cavitation discharge": "出口汽蚀",
    "Bearing BPFI": "轴承内圈故障",
    "Bearing BPFO": "轴承外圈故障",
    "Bearing BSF": "轴承滚动体故障",
    "Bearing contaminated": "轴承污染",
    "Bearing pump": "泵轴承故障",
    "Pump bearing": "泵轴承故障",
    "Pump unbalance": "泵不平衡",
    "Motor unbalance": "电机不平衡",
    "Unbalance pump": "泵不平衡",
    "Unbalance motor": "电机不平衡",
    "Angular misalignment": "角向不对中",
    "Parallel misalignment": "平行不对中",
    "Combined misalignment": "组合不对中",
    "Align angular": "角向不对中",
    "Align parallel": "平行不对中",
    "Align combination": "组合不对中",
    "Soft foot": "软脚",
    "Motor base looseness": "电机地脚松动",
    "Pump base looseness": "泵地脚松动",
    "Loose foot motor": "电机地脚松动",
    "Loose foot pump": "泵地脚松动",
    "Coupling": "联轴器故障",
    "Impeller": "叶轮故障",
    "Broken rotor bar": "转子断条",
    "New motor": "新电机",
    "Bent shaft": "弯轴",
    "Stator short": "定子短路",
}


@dataclass(frozen=True)
class ConditionCatalog:
    rpm_by_key: dict[tuple[str, int, str], float]
    nominal_rpm_by_speed: dict[tuple[str, int], float]

    @classmethod
    def from_workbook(cls, workbook_path: Path) -> "ConditionCatalog":
        frame = pd.read_excel(workbook_path, sheet_name="Ordered Measurements", header=1)
        frame = frame.rename(columns=_WORKBOOK_COLUMN_RENAMES)
        required = {"setup", "failure_description", "speed_percent", "speed_rpm"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"工况表缺少列: {sorted(missing)}")

        frame = frame.loc[
            frame["setup"].astype(str).str.lower().ne("setup")
            & frame["failure_description"].astype(str).str.lower().ne("failure description")
        ].copy()

        rpm_by_key: dict[tuple[str, int, str], float] = {}
        rpm_by_speed: dict[tuple[str, int], list[float]] = {}
        selected = ["setup", "failure_description", "speed_percent", "speed_rpm"]
        if "Severity" in frame.columns:
            selected.append("Severity")
        for row in frame.loc[:, selected].dropna(subset=["setup", "failure_description", "speed_percent", "speed_rpm"]).itertuples(index=False):
            machine_id = _normalize_machine_id(str(row.setup))
            speed_percent = _normalize_speed_percent(row.speed_percent)
            severity = getattr(row, "Severity", None)
            raw_fault = _normalize_fault_name(str(row.failure_description), severity)
            rpm = _normalize_rpm(row.speed_rpm)
            key = (machine_id, speed_percent, raw_fault)
            if key in rpm_by_key and not np.isclose(rpm_by_key[key], rpm):
                raise ValueError(f"冲突转速: {key}")
            rpm_by_key[key] = rpm
            rpm_by_speed.setdefault((machine_id, speed_percent), []).append(rpm)

        nominal_rpm_by_speed: dict[tuple[str, int], float] = {}
        for key, values in rpm_by_speed.items():
            counts = Counter(values)
            max_count = max(counts.values())
            candidates = [rpm for rpm, count in counts.items() if count == max_count]
            nominal_rpm_by_speed[key] = float(min(candidates))

        return cls(rpm_by_key=rpm_by_key, nominal_rpm_by_speed=nominal_rpm_by_speed)

    def rpm_for(self, machine_id: str, speed_percent: int, raw_fault: str) -> float | None:
        return self.rpm_by_key.get((machine_id, speed_percent, raw_fault))

    def nominal_rpm_for(self, machine_id: str, speed_percent: int) -> float | None:
        return self.nominal_rpm_by_speed.get((machine_id, speed_percent))


def build_file_index(
    config: PipelineConfig,
    *,
    catalog: ConditionCatalog | None = None,
    rpm_lookup: Mapping[tuple[str, int, str], float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(config.data_root.rglob("*通道4.csv")):
        raw_fault = csv_path.parent.name
        label = map_fault_to_label(raw_fault)
        if label is None:
            continue

        frame = pd.read_csv(csv_path)
        if "time" not in frame.columns:
            raise ValueError(f"CSV缺少time列: {csv_path}")

        time_values = frame["time"].to_numpy(dtype=np.float64)
        original_fs = _infer_sampling_rate(time_values)
        signal_columns = [column for column in frame.columns if column != "time"]
        machine_id, speed_percent = _parse_machine_and_speed(csv_path)
        rpm = _resolve_rpm(machine_id, speed_percent, raw_fault, catalog, rpm_lookup)
        file_hash = _sha256(csv_path)

        for run_index, signal_column in enumerate(signal_columns):
            samples = frame[signal_column].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "file_path": str(csv_path),
                    "label": label,
                    "raw_fault": raw_fault,
                    "speed_percent": speed_percent,
                    "rpm": rpm,
                    "machine_id": machine_id,
                    "condition_id": f"{machine_id}_{speed_percent}_{raw_fault}",
                    "run_id": f"{csv_path.stem}_run_{run_index}",
                    "source_column": str(signal_column),
                    "channel": config.channel,
                    "original_fs": original_fs,
                    "n_samples": int(samples.shape[0]),
                    "duration_s": float(samples.shape[0] / original_fs),
                    "status": "indexed",
                    "sha256": file_hash,
                }
            )
    return pd.DataFrame(rows)


def split_file_index(
    index: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    grouped = index.loc[:, ["file_path", "label", "sha256"]].drop_duplicates().reset_index(drop=True)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(splitter.split(grouped["file_path"], grouped["label"]))
    train_files = set(grouped.iloc[train_idx]["file_path"])
    test_files = set(grouped.iloc[test_idx]["file_path"])
    train_rows = index[index["file_path"].isin(train_files)].reset_index(drop=True)
    test_rows = index[index["file_path"].isin(test_files)].reset_index(drop=True)
    report = {
        "source_file_overlap_count": len(train_files.intersection(test_files)),
        "hash_overlap_count": len(set(train_rows["sha256"]).intersection(test_rows["sha256"])),
    }
    return train_rows, test_rows, report


def _normalize_machine_id(value: str) -> str:
    digits = re.findall(r"\d+", value)
    if not digits:
        raise ValueError(f"无法解析电机编号: {value}")
    return f"Motor-{int(digits[0])}"


def _normalize_speed_percent(value: object) -> int:
    speed = _coerce_last_numeric(value)
    if speed <= 1.0:
        speed *= 100.0
    return int(round(speed))


def _normalize_fault_name(value: str, severity: object | None = None) -> str:
    text = " ".join(value.strip().split())
    match = re.search(r"(\d+)$", text)
    severity_suffix = match.group(1) if match else ""
    base = text[: match.start()].strip() if match else text
    chinese_base = _WORKBOOK_FAILURE_MAP.get(base, base)
    missing_severity = severity is None or pd.isna(severity) or str(severity).strip() in {"", "N/A", "nan", "NaN"}
    if chinese_base == "正常状态" and not severity_suffix and missing_severity:
        return f"{chinese_base}1"
    if not severity_suffix and not missing_severity:
        severity_text = str(severity).strip()
        try:
            severity_suffix = str(int(float(severity_text)))
        except ValueError:
            severity_suffix = severity_text
    return f"{chinese_base}{severity_suffix}"


def _normalize_rpm(value: object) -> float:
    return float(_coerce_last_numeric(value))


def _coerce_last_numeric(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    matches = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        raise ValueError(f"无法解析数值: {value}")
    return float(matches[-1])


def _infer_sampling_rate(time_values: np.ndarray) -> int:
    if time_values.shape[0] < 2:
        raise ValueError("time列长度不足以推断采样率")
    delta = float(np.median(np.diff(time_values)))
    if delta <= 0:
        raise ValueError("time列步长无效")
    return int(round(1.0 / delta))


def _parse_machine_and_speed(csv_path: Path) -> tuple[str, int]:
    parts = csv_path.parts
    if len(parts) < 3:
        raise ValueError(f"路径层级不足: {csv_path}")
    machine_id = _normalize_machine_id(parts[-4])
    speed_percent = int(parts[-3])
    return machine_id, speed_percent


def _resolve_rpm(
    machine_id: str,
    speed_percent: int,
    raw_fault: str,
    catalog: ConditionCatalog | None,
    rpm_lookup: Mapping[tuple[str, int, str], float] | None,
) -> float | None:
    key = (machine_id, speed_percent, raw_fault)
    if rpm_lookup is not None and key in rpm_lookup:
        return float(rpm_lookup[key])
    if catalog is not None:
        nominal = catalog.nominal_rpm_for(machine_id, speed_percent)
        if nominal is not None:
            return nominal
        return catalog.rpm_for(machine_id, speed_percent, raw_fault)
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
