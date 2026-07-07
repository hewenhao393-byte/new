from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pump_fault_app.io.signal_reader import (
    RawSignalReadError,
    RawSignalReadRequest,
    read_vibration_signal,
)


def test_read_single_column_csv(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("signal\n1\n2\n3\n4\n", encoding="utf-8")

    record = read_vibration_signal(
        RawSignalReadRequest(
            file_path=path,
            sampling_rate_hz=12_000,
            rpm=1480.0,
        )
    )

    assert record.file_name == "single.csv"
    assert record.sample_count == 4
    assert record.selected_signal_column == "signal"
    assert np.allclose(record.samples, np.array([1.0, 2.0, 3.0, 4.0]))


def test_read_csv_with_time_and_auto_pick_channel4(tmp_path: Path) -> None:
    path = tmp_path / "channels.csv"
    path.write_text(
        "time,通道1,通道4\n0.0,0.1,1.1\n0.1,0.2,1.2\n0.2,0.3,1.3\n",
        encoding="utf-8",
    )

    record = read_vibration_signal(
        RawSignalReadRequest(
            file_path=path,
            sampling_rate_hz=20_000,
            rpm=2070.0,
        )
    )

    assert record.selected_signal_column == "通道4"
    assert record.time_column == "time"
    assert np.allclose(record.samples, np.array([1.1, 1.2, 1.3]))


def test_read_multi_channel_csv_with_user_selected_column(tmp_path: Path) -> None:
    path = tmp_path / "multi.csv"
    path.write_text(
        "time,ch1,ch2,ch3,ch4\n0.0,1,2,3,4\n0.1,5,6,7,8\n",
        encoding="utf-8",
    )

    record = read_vibration_signal(
        RawSignalReadRequest(
            file_path=path,
            sampling_rate_hz=12_000,
            rpm=1480.0,
            signal_column="ch2",
            time_column="time",
            device_id="Motor-2",
            measurement_position="pump_side",
        )
    )

    assert record.selected_signal_column == "ch2"
    assert record.time_column == "time"
    assert record.device_id == "Motor-2"
    assert record.measurement_position == "pump_side"
    assert np.allclose(record.samples, np.array([2.0, 6.0]))


def test_read_txt_numeric_file(tmp_path: Path) -> None:
    path = tmp_path / "signal.txt"
    path.write_text("说明行\n1.0\n2.5\n3.5\n", encoding="utf-8")

    record = read_vibration_signal(
        RawSignalReadRequest(
            file_path=path,
            sampling_rate_hz=10_000,
            rpm=1450.0,
        )
    )

    assert record.selected_signal_column == "column_0"
    assert np.allclose(record.samples, np.array([1.0, 2.5, 3.5]))


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(RawSignalReadError, match="file is empty"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=12_000,
                rpm=1480.0,
            )
        )


def test_missing_numeric_columns_raise(tmp_path: Path) -> None:
    path = tmp_path / "text_only.csv"
    path.write_text("name,desc\na,b\nc,d\n", encoding="utf-8")

    with pytest.raises(RawSignalReadError, match="no numeric signal column"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=12_000,
                rpm=1480.0,
            )
        )


def test_selected_column_all_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty_col.csv"
    path.write_text("time,ch4\n0.0,\n0.1,\n", encoding="utf-8")

    with pytest.raises(RawSignalReadError, match="selected signal column is empty"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=12_000,
                rpm=1480.0,
                signal_column="ch4",
            )
        )


def test_missing_sampling_rate_raises(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("signal\n1\n2\n", encoding="utf-8")

    with pytest.raises(RawSignalReadError, match="sampling rate is required"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=None,
                rpm=1480.0,
            )
        )


def test_missing_rpm_raises(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("signal\n1\n2\n", encoding="utf-8")

    with pytest.raises(RawSignalReadError, match="rpm is required"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=12_000,
                rpm=None,
            )
        )


def test_unsupported_file_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "signal.bin"
    path.write_bytes(b"\x00\x01\x02")

    with pytest.raises(RawSignalReadError, match="unsupported file format"):
        read_vibration_signal(
            RawSignalReadRequest(
                file_path=path,
                sampling_rate_hz=12_000,
                rpm=1480.0,
            )
        )
