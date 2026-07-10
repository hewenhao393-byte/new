from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pump_fault_app.batch import BatchInferenceResult


@dataclass(frozen=True)
class BatchExportResult:
    output_dir: Path
    json_path: Path
    csv_path: Path


@dataclass(frozen=True)
class SummaryExportResult:
    output_dir: Path
    json_path: Path
    csv_path: Path


def create_export_output_dir(base_dir: Path, *, prefix: str = "batch_run", timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{prefix}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_batch_result(batch_result: BatchInferenceResult, *, output_dir: Path) -> BatchExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_diagnosis_summary.json"
    csv_path = output_dir / "batch_diagnosis_summary.csv"

    json_path.write_text(
        json.dumps(asdict(batch_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(batch_result, csv_path)
    return BatchExportResult(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
    )


def export_diagnosis_summary(summary: object, *, output_dir: Path) -> SummaryExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnosis_summary.json"
    csv_path = output_dir / "diagnosis_summary.csv"
    summary_dict = summary.as_dict()
    json_path.write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_dict.keys()))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
                for key, value in summary_dict.items()
            }
        )
    return SummaryExportResult(output_dir=output_dir, json_path=json_path, csv_path=csv_path)


def _write_summary_csv(batch_result: BatchInferenceResult, csv_path: Path) -> None:
    fieldnames = [
        "success",
        "status",
        "message",
        "file_name",
        "sampling_rate_hz",
        "rpm",
        "device_id",
        "measurement_position",
        "diagnosis_label",
        "confidence",
        "window_count",
        "failure_stage",
        "failure_message",
        "warnings",
        "rejection_reasons",
        "top_probabilities",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in batch_result.summaries:
            writer.writerow(
                {
                    "success": summary.success,
                    "status": summary.status,
                    "message": summary.message,
                    "file_name": summary.file_name,
                    "sampling_rate_hz": summary.sampling_rate_hz,
                    "rpm": summary.rpm,
                    "device_id": summary.device_id,
                    "measurement_position": summary.measurement_position,
                    "diagnosis_label": summary.diagnosis_label,
                    "confidence": summary.confidence,
                    "window_count": summary.window_count,
                    "failure_stage": summary.failure_stage,
                    "failure_message": summary.failure_message,
                    "warnings": " | ".join(summary.warnings),
                    "rejection_reasons": " | ".join(summary.rejection_reasons),
                    "top_probabilities": json.dumps(summary.top_probabilities, ensure_ascii=False),
                }
            )
