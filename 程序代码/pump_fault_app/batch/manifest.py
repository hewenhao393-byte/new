from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pump_fault_app.batch.service import BatchInferenceRequest, BatchInferenceResult, run_batch_inference
from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest


_REQUIRED_COLUMNS = ("sampling_rate_hz", "rpm")


@dataclass(frozen=True)
class BatchManifestItem:
    request: MultiChannelInferenceRequest


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
    model_directory: Path | None = None,
) -> BatchInferenceResult:
    requests = tuple(
        MultiChannelInferenceRequest(
            channels=item.request.channels,
            sampling_rate_hz=item.request.sampling_rate_hz,
            rpm=item.request.rpm,
            model_directory=model_directory or item.request.model_directory,
        )
        for item in manifest.items
    )
    return run_batch_inference(BatchInferenceRequest(requests))


def _row_to_manifest_item(row: pd.Series) -> BatchManifestItem:
    row_number = int(row.name) + 1
    sampling_rate_hz = int(row["sampling_rate_hz"])
    if sampling_rate_hz <= 0:
        raise ValueError(f"manifest row {row_number} has invalid sampling_rate_hz")
    rpm = float(row["rpm"])
    if rpm <= 0.0:
        raise ValueError(f"manifest row {row_number} has invalid rpm")

    channels = []
    for channel in ("CH3", "CH4", "CH5"):
        prefix = channel.lower()
        file_path = _optional_text(row, f"{prefix}_file")
        if file_path is None:
            continue
        channels.append(
            ChannelInput(
                channel,
                Path(file_path),
                _optional_text(row, f"{prefix}_signal_column"),
                _optional_text(row, f"{prefix}_time_column"),
            )
        )
    if not channels:
        raise ValueError(f"manifest row {row_number} must provide at least one channel file")
    return BatchManifestItem(
        MultiChannelInferenceRequest(tuple(channels), sampling_rate_hz=sampling_rate_hz, rpm=rpm)
    )


def _optional_text(row: pd.Series, column: str) -> str | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
