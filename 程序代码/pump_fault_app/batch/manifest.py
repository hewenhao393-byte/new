from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pump_fault_app.batch.service import BatchInferenceResult
from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference
from pump_fault_app.reporting import build_diagnosis_summary


_REQUIRED_COLUMNS = ("file_path", "sampling_rate_hz", "rpm")


@dataclass(frozen=True)
class BatchManifestItem:
    file_path: Path
    sampling_rate_hz: int
    rpm: float
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None


@dataclass(frozen=True)
class BatchManifest:
    source_path: Path
    items: tuple[BatchManifestItem, ...]


def load_batch_manifest(manifest_path: Path | str) -> BatchManifest:
    path = Path(manifest_path)
    frame = pd.read_csv(path)
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("manifest must contain at least one row")

    items = tuple(_row_to_manifest_item(row) for _, row in frame.iterrows())
    return BatchManifest(source_path=path, items=items)


def run_batch_inference_from_manifest(
    manifest: BatchManifest,
    *,
    model_bundle_path: Path | None = None,
) -> BatchInferenceResult:
    summaries = tuple(
        build_diagnosis_summary(
            run_formal_inference(
                FormalInferenceRequest(
                    file_path=item.file_path,
                    sampling_rate_hz=item.sampling_rate_hz,
                    rpm=item.rpm,
                    signal_column=item.signal_column,
                    time_column=item.time_column,
                    device_id=item.device_id,
                    measurement_position=item.measurement_position,
                    model_bundle_path=model_bundle_path,
                )
            )
        )
        for item in manifest.items
    )
    diagnosed_count = sum(summary.status == "diagnosed" for summary in summaries)
    rejected_count = sum(summary.status == "rejected" for summary in summaries)
    input_error_count = sum(summary.status == "input_error" for summary in summaries)
    success_count = sum(summary.success for summary in summaries)
    total_count = len(summaries)
    return BatchInferenceResult(
        total_count=total_count,
        success_count=success_count,
        failure_count=total_count - success_count,
        diagnosed_count=diagnosed_count,
        rejected_count=rejected_count,
        input_error_count=input_error_count,
        summaries=summaries,
    )


def _row_to_manifest_item(row: pd.Series) -> BatchManifestItem:
    row_number = int(row.name) + 1
    file_path_text = str(row["file_path"]).strip()
    if not file_path_text:
        raise ValueError(f"manifest row {row_number} has empty file_path")
    sampling_rate_hz = int(row["sampling_rate_hz"])
    if sampling_rate_hz <= 0:
        raise ValueError(f"manifest row {row_number} has invalid sampling_rate_hz")
    rpm = float(row["rpm"])
    if rpm <= 0.0:
        raise ValueError(f"manifest row {row_number} has invalid rpm")

    return BatchManifestItem(
        file_path=Path(file_path_text),
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        signal_column=_optional_text(row, "signal_column"),
        time_column=_optional_text(row, "time_column"),
        device_id=_optional_text(row, "device_id"),
        measurement_position=_optional_text(row, "measurement_position"),
    )


def _optional_text(row: pd.Series, column: str) -> str | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
