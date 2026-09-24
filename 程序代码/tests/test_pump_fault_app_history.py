from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from pump_fault_app.history import DiagnosisHistoryRecord, SQLiteDiagnosisHistoryRepository
from pump_fault_app.history.deletion import delete_history_record
from pump_fault_app.sample_repository import SQLiteSampleRepository
from pump_fault_app.presentation.history import (
    build_history_detail,
    build_history_select_options,
    build_history_summary,
    build_history_table_rows,
)
from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def _history_record(
    *,
    diagnosed_at: str,
    file_name: str,
    report_path: Path,
) -> DiagnosisHistoryRecord:
    return DiagnosisHistoryRecord(
        diagnosed_at=diagnosed_at,
        file_name=file_name,
        sampling_rate_hz=20000,
        rpm=2070.0,
        predicted_label="汽蚀",
        confidence=0.96,
        window_consistency=0.875,
        report_path=report_path,
    )


def test_sqlite_history_repository_creates_schema_and_round_trips_records(tmp_path: Path) -> None:
    database_path = tmp_path / "history.sqlite3"
    repository = SQLiteDiagnosisHistoryRepository(database_path)
    old_report = (tmp_path / "old.docx").resolve()
    new_report = (tmp_path / "new.docx").resolve()

    old = repository.add(
        _history_record(
            diagnosed_at="2026-07-28 20:00:00",
            file_name="old.csv",
            report_path=old_report,
        )
    )
    newer = repository.add(
        _history_record(
            diagnosed_at="2026-07-28 20:30:00",
            file_name="new.csv",
            report_path=new_report,
        )
    )

    assert old.id is not None
    assert newer.id is not None
    assert repository.get(newer.id) == newer
    assert [item.file_name for item in repository.list_all()] == ["new.csv", "old.csv"]
    assert newer.sampling_rate_hz == 20000
    assert newer.rpm == 2070.0
    assert newer.predicted_label == "汽蚀"
    assert newer.confidence == 0.96
    assert newer.window_consistency == 0.875
    assert newer.report_path == new_report

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(diagnosis_history)").fetchall()
        }
    assert columns == {
        "id",
        "diagnosed_at",
        "file_name",
        "sampling_rate_hz",
        "rpm",
        "predicted_label",
        "confidence",
        "window_consistency",
        "report_path",
    }


def test_sqlite_history_repository_returns_none_for_unknown_record(tmp_path: Path) -> None:
    repository = SQLiteDiagnosisHistoryRepository(tmp_path / "history.sqlite3")

    assert repository.get(999) is None


def test_sqlite_history_repository_deletes_only_the_selected_record(tmp_path: Path) -> None:
    repository = SQLiteDiagnosisHistoryRepository(tmp_path / "history.sqlite3")
    first = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:00:00",
            file_name="first.csv",
            report_path=tmp_path / "first.docx",
        )
    )
    second = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:01:00",
            file_name="second.csv",
            report_path=tmp_path / "second.docx",
        )
    )

    assert repository.delete(int(first.id)) is True
    assert repository.get(int(first.id)) is None
    assert repository.get(int(second.id)) == second
    assert repository.delete(999) is False


def test_history_deletion_service_removes_report_and_record(tmp_path: Path) -> None:
    repository = SQLiteDiagnosisHistoryRepository(tmp_path / "history.sqlite3")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    report_path = report_root / "pump.docx"
    report_path.write_bytes(b"report")
    record = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:00:00",
            file_name="pump.csv",
            report_path=report_path,
        )
    )

    result = delete_history_record(repository, record, report_root=report_root)

    assert result.status == "deleted"
    assert not report_path.exists()
    assert repository.get(int(record.id)) is None


def test_history_deletion_service_handles_missing_and_protected_reports(tmp_path: Path) -> None:
    repository = SQLiteDiagnosisHistoryRepository(tmp_path / "history.sqlite3")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    missing = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:00:00",
            file_name="missing.csv",
            report_path=report_root / "missing.docx",
        )
    )
    protected_path = report_root / "protected.docx"
    protected_path.write_bytes(b"report")
    protected = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:01:00",
            file_name="protected.csv",
            report_path=protected_path,
        )
    )

    missing_result = delete_history_record(repository, missing, report_root=report_root)
    failed_result = delete_history_record(
        repository,
        protected,
        report_root=report_root,
        remove_file=lambda _: (_ for _ in ()).throw(PermissionError("locked")),
    )

    assert missing_result.status == "record_deleted_report_missing"
    assert repository.get(int(missing.id)) is None
    assert failed_result.status == "report_delete_failed"
    assert repository.get(int(protected.id)) == protected
    assert protected_path.exists()


def test_history_deletion_service_rejects_report_path_outside_system_directory(tmp_path: Path) -> None:
    repository = SQLiteDiagnosisHistoryRepository(tmp_path / "history.sqlite3")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    outside_report = tmp_path / "outside.docx"
    outside_report.write_bytes(b"report")
    record = repository.add(
        _history_record(
            diagnosed_at="2026-08-04 10:00:00",
            file_name="outside.csv",
            report_path=outside_report,
        )
    )

    result = delete_history_record(repository, record, report_root=report_root)

    assert result.status == "report_path_outside_system_directory"
    assert outside_report.exists()
    assert repository.get(int(record.id)) == record


def test_successful_single_diagnosis_saves_word_report_and_history_record(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)
    signal_path = tmp_path / "pump_record.csv"
    _write_signal_csv(
        signal_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    database_path = tmp_path / "history.sqlite3"
    sample_database_path = tmp_path / "samples.sqlite3"

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            history_database_path=database_path,
            history_report_dir=tmp_path / "reports",
            sample_database_path=sample_database_path,
        )
    )

    assert result.summary.success is True
    assert result.history_warning is None
    assert result.history_record is not None
    assert result.history_record.file_name == "pump_record.csv"
    assert result.history_record.sampling_rate_hz == 12000
    assert result.history_record.rpm == 1500.0
    assert result.history_record.predicted_label == result.summary.diagnosis_label
    assert result.history_record.confidence == result.summary.confidence
    assert 0.0 <= result.history_record.window_consistency <= 1.0
    assert result.history_record.report_path.exists()
    assert result.history_record.report_path.suffix == ".docx"
    assert SQLiteDiagnosisHistoryRepository(database_path).get(result.history_record.id) == result.history_record
    assert SQLiteSampleRepository(sample_database_path).snapshot_count(
        history_record_id=int(result.history_record.id)
    ) == result.inference_result.record_prediction.window_count


def test_rejected_single_diagnosis_does_not_create_history_record(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)
    signal_path = tmp_path / "constant.csv"
    _write_signal_csv(signal_path, np.ones(4800, dtype=np.float64))
    database_path = tmp_path / "history.sqlite3"

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            history_database_path=database_path,
            history_report_dir=tmp_path / "reports",
        )
    )

    assert result.summary.success is False
    assert result.history_record is None
    assert result.history_warning is None
    assert not database_path.exists()


def test_history_storage_failure_does_not_change_successful_prediction(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)
    signal_path = tmp_path / "pump_record.csv"
    _write_signal_csv(
        signal_path,
        0.8 * np.sin(2.0 * np.pi * 25.0 * np.arange(4800) / 12000.0),
    )
    invalid_database_path = tmp_path / "database_is_a_directory"
    invalid_database_path.mkdir()

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=signal_path,
            sampling_rate_hz=12000,
            rpm=1500.0,
            model_bundle_path=bundle_path,
            history_database_path=invalid_database_path,
            history_report_dir=tmp_path / "reports",
        )
    )

    assert result.summary.success is True
    assert result.summary.diagnosis_label is not None
    assert result.summary.confidence is not None
    assert result.history_record is None
    assert result.history_warning == "诊断已完成，但历史记录或Word报告未能保存。"


def test_history_presentation_formats_table_summary_and_detail(tmp_path: Path) -> None:
    report_path = tmp_path / "pump.docx"
    report_path.write_bytes(b"docx")
    record = DiagnosisHistoryRecord(
        id=2,
        diagnosed_at="2026-07-28 20:30:00",
        file_name="pump.csv",
        sampling_rate_hz=20000,
        rpm=2070.0,
        predicted_label="汽蚀",
        confidence=0.96,
        window_consistency=0.875,
        report_path=report_path,
    )

    assert build_history_table_rows([record]) == [
        {
            "记录ID": 2,
            "诊断时间": "2026-07-28 20:30:00",
            "文件名": "pump.csv",
            "采样率": "20000 Hz",
            "转速": "2070.0 rpm",
            "预测类别": "汽蚀",
            "置信度": "96.0%",
            "窗口一致率": "87.5%",
            "报告状态": "可打开",
        }
    ]
    assert build_history_summary([record]) == {
        "历史记录数": "1",
        "最近诊断时间": "2026-07-28 20:30:00",
    }
    assert build_history_detail(record)["Word报告路径"] == str(report_path)
    assert build_history_select_options([record]) == {
        2: "2026-07-28 20:30:00｜pump.csv｜汽蚀"
    }


def test_history_presentation_marks_missing_report_without_hiding_record(tmp_path: Path) -> None:
    record = _history_record(
        diagnosed_at="2026-07-28 20:30:00",
        file_name="missing.csv",
        report_path=tmp_path / "missing.docx",
    )

    assert build_history_table_rows([record])[0]["报告状态"] == "报告文件不存在"
    assert build_history_detail(record)["报告状态"] == "报告文件不存在"
