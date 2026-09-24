from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pump_fault_app.domain.records import RawSignalRecord


class RawSignalReadError(ValueError):
    pass


@dataclass(frozen=True)
class RawSignalReadRequest:
    file_path: Path
    sampling_rate_hz: int | None
    rpm: float | None
    signal_column: str | None = None
    time_column: str | None = None
    device_id: str | None = None
    measurement_position: str | None = None


def read_vibration_signal(request: RawSignalReadRequest) -> RawSignalRecord:
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise RawSignalReadError(f"file does not exist: {file_path}")
    if file_path.stat().st_size == 0:
        raise RawSignalReadError("file is empty")
    if request.sampling_rate_hz is None:
        raise RawSignalReadError("sampling rate is required")
    if request.rpm is None:
        raise RawSignalReadError("rpm is required")

    suffix = file_path.suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise RawSignalReadError("unsupported file format")

    frame = _load_table(file_path)
    selected_signal_column, time_column = _select_columns(frame, request.signal_column, request.time_column)
    numeric_signal = pd.to_numeric(frame[selected_signal_column], errors="coerce")
    if not numeric_signal.notna().any():
        raise RawSignalReadError("selected signal column is empty")
    samples = numeric_signal.to_numpy(dtype=np.float64)

    sampling_rate_hz = int(request.sampling_rate_hz)
    duration_seconds = float(samples.size / sampling_rate_hz)
    notes: list[str] = []
    if request.signal_column is None and selected_signal_column.lower() not in {"通道4", "channel4", "channel_4", "ch4"}:
        notes.append("auto-selected signal column")

    return RawSignalRecord(
        file_name=file_path.name,
        source_file=file_path,
        samples=samples,
        sample_count=int(samples.size),
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        rpm=float(request.rpm),
        device_id=request.device_id,
        measurement_position=request.measurement_position,
        selected_signal_column=selected_signal_column,
        time_column=time_column,
        notes=tuple(notes),
    )


def _load_table(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".txt":
        readers = (_read_text_numeric_lines, _read_delimited_with_header, _read_delimited_without_header)
    else:
        readers = (_read_delimited_with_header, _read_delimited_without_header, _read_text_numeric_lines)
    for reader in readers:
        frame = reader(file_path)
        if frame is not None and not frame.empty:
            return frame
    raise RawSignalReadError("no numeric signal column could be read from file")


def _read_delimited_with_header(file_path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(file_path, on_bad_lines="skip")
    except Exception:
        try:
            frame = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            return None
    return _normalize_frame(frame)


def _read_delimited_without_header(file_path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip", header=None)
    except Exception:
        return None
    frame.columns = [f"column_{index}" for index in range(frame.shape[1])]
    return _normalize_frame(frame)


def _read_text_numeric_lines(file_path: Path) -> pd.DataFrame | None:
    rows: list[list[float]] = []
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = [token for token in stripped.replace(",", " ").split() if token]
        if not tokens:
            continue
        try:
            numeric_row = [float(token) for token in tokens]
        except ValueError:
            continue
        rows.append(numeric_row)
    if not rows:
        return None
    width = max(len(row) for row in rows)
    padded = [row + [np.nan] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=[f"column_{index}" for index in range(width)])


def _normalize_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    normalized = frame.copy()
    normalized = normalized.dropna(axis=0, how="all")
    if normalized.empty:
        return None
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def _select_columns(
    frame: pd.DataFrame,
    requested_signal_column: str | None,
    requested_time_column: str | None,
) -> tuple[str, str | None]:
    time_column = _detect_time_column(frame, requested_time_column)
    if requested_signal_column is not None:
        if requested_signal_column not in frame.columns:
            raise RawSignalReadError(f"requested signal column not found: {requested_signal_column}")
        return requested_signal_column, time_column

    numeric_columns = _numeric_columns(frame)
    if not numeric_columns:
        raise RawSignalReadError("no numeric signal column found")

    signal_candidates = [column for column in numeric_columns if column != time_column]
    if not signal_candidates:
        raise RawSignalReadError("no numeric signal column found")

    for preferred in ("通道4", "channel4", "channel_4", "ch4"):
        for column in signal_candidates:
            if column.strip().lower() == preferred:
                return column, time_column

    counts = {
        column: int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
        for column in signal_candidates
    }
    selected = max(signal_candidates, key=lambda name: (counts[name], -list(frame.columns).index(name)))
    return selected, time_column


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns: list[str] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            numeric_columns.append(str(column))
    return numeric_columns


def _detect_time_column(frame: pd.DataFrame, requested_time_column: str | None = None) -> str | None:
    if requested_time_column is not None:
        if requested_time_column not in frame.columns:
            raise RawSignalReadError(f"requested time column not found: {requested_time_column}")
        return requested_time_column
    for column in frame.columns:
        lowered = str(column).strip().lower()
        if lowered in {"time", "timestamp", "t"}:
            return str(column)
    return None
