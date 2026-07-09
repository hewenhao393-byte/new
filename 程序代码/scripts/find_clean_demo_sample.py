#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pump_fault_app.reporting.view_data import build_single_report_view_data
from pump_fault_app.services.app_service import AppSingleRunRequest, run_single_diagnosis
from pump_fault_app.services.report_export_service import export_single_diagnosis_report

DEFAULT_DATASET_PATH = PROJECT_ROOT.parent / "实验结果" / "多转速统一六分类实验V2" / "unified_six_class_dataset_raw.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "demo_outputs" / "demo_sample_candidates.csv"

CANDIDATE_PRIORITY = (
    ("Motor-4", 70, "汽蚀", 1),
    ("Motor-2", 100, "轴承故障", 2),
    ("Motor-2", 100, "松动", 3),
    ("Motor-4", 70, "转子不平衡", 4),
    ("Motor-4", 70, "联轴器不对中", 5),
    ("Motor-2", 100, "正常", 6),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="筛选更适合答辩演示的正式单文件样本。")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="unified_six_class_dataset_raw.csv 路径",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="筛选结果 CSV 输出路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="输出前 N 条候选样本",
    )
    return parser


def _candidate_rows(dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path, usecols=["device_id", "speed_percent", "rpm", "label", "source_file"])
    frame = frame.drop_duplicates().copy()
    allowed = {(device_id, speed, label): order for device_id, speed, label, order in CANDIDATE_PRIORITY}
    frame["priority_order"] = frame.apply(
        lambda row: allowed.get((row["device_id"], int(row["speed_percent"]), row["label"])),
        axis=1,
    )
    return frame[frame["priority_order"].notna()].copy()


def _make_request(row: pd.Series) -> AppSingleRunRequest:
    return AppSingleRunRequest(
        file_path=Path(str(row["source_file"])),
        sampling_rate_hz=20000,
        rpm=float(row["rpm"]),
        signal_column="0",
        time_column="time",
        device_id=str(row["device_id"]),
        measurement_position="通道4",
    )


def _evaluate_candidate(row: pd.Series) -> dict[str, object]:
    request = _make_request(row)
    result = run_single_diagnosis(request)
    summary = result.summary
    view_data = build_single_report_view_data(result)
    conclusion = view_data.to_dict()["conclusion"]
    visualization_complete = all(
        [
            result.visualization is not None,
            result.visualization.time_domain is not None if result.visualization else False,
            result.visualization.frequency_spectrum is not None if result.visualization else False,
            result.visualization.envelope_spectrum is not None if result.visualization else False,
            result.visualization.wavelet_packet_energy is not None if result.visualization else False,
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "candidate_report.docx"
        export_single_diagnosis_report(result, report_path)
        word_export_ok = report_path.exists() and report_path.stat().st_size > 0

    return {
        "priority_order": int(row["priority_order"]),
        "device_id": str(row["device_id"]),
        "speed_percent": int(row["speed_percent"]),
        "file_path": str(row["source_file"]),
        "expected_label": str(row["label"]),
        "predicted_label": str(summary.diagnosis_label or "-"),
        "status": str(summary.status),
        "confidence": float(summary.confidence or 0.0),
        "second_label": str(conclusion["second_label"] or "-"),
        "probability_margin": float(conclusion["probability_margin"] or 0.0),
        "window_count": int(summary.window_count or 0),
        "window_consistency": float(conclusion["window_consistency"] or 0.0),
        "warning_count": int(len(summary.runtime_alerts)),
        "visualization_complete": bool(visualization_complete),
        "word_export_ok": bool(word_export_ok),
    }


def _candidate_sort_key(item: dict[str, object]) -> tuple[object, ...]:
    return (
        int(item["warning_count"]),
        -float(item["confidence"]),
        -float(item["probability_margin"]),
        -float(item["window_consistency"]),
        int(item["priority_order"]),
        str(item["file_path"]),
    )


def run_selection(dataset_path: Path, output_csv: Path, top_k: int) -> list[dict[str, object]]:
    candidates = _candidate_rows(dataset_path)
    rows = [_evaluate_candidate(row) for _, row in candidates.iterrows()]
    rows.sort(key=_candidate_sort_key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows[:top_k]


def main() -> int:
    args = build_parser().parse_args()
    top_rows = run_selection(args.dataset, args.output_csv, args.top_k)

    print(f"Top {len(top_rows)} demo sample candidates")
    for index, row in enumerate(top_rows, start=1):
        print(
            f"{index:02d}. {row['expected_label']} -> {row['predicted_label']} | "
            f"confidence={row['confidence']:.4f} | margin={row['probability_margin']:.4f} | "
            f"consistency={row['window_consistency']:.4f} | warnings={row['warning_count']} | {row['file_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
