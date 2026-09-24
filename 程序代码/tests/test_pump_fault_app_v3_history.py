from __future__ import annotations

from pathlib import Path

from pump_fault_app.domain.diagnosis_models import ChannelDiagnosisResult
from pump_fault_app.fusion import fuse_valid_channels
from pump_fault_app.history import V3HistoryRepository


def _diagnosed_result():
    return fuse_valid_channels(
        (
            ChannelDiagnosisResult(
                channel="CH3",
                status="valid",
                predicted_label="正常",
                class_probabilities=(0.7, 0.1, 0.05, 0.05, 0.05, 0.05),
            ),
            ChannelDiagnosisResult(
                channel="CH4",
                status="valid",
                predicted_label="正常",
                class_probabilities=(0.6, 0.1, 0.1, 0.05, 0.1, 0.05),
            ),
        )
    )


def test_v3_history_saves_channels_probabilities_and_versions(tmp_path: Path) -> None:
    repo = V3HistoryRepository(tmp_path / "history_v3.sqlite3")
    record = repo.save(
        _diagnosed_result(),
        source_files={"CH3": "a.csv", "CH4": "b.csv"},
        sampling_rate_hz=20_000,
        rpm=1500.0,
    )
    loaded = repo.get(int(record.id))
    assert loaded is not None
    assert loaded.input_channels == ("CH3", "CH4")
    assert loaded.model_version == "catboost43-six-class-v3"
    assert loaded.feature_version == "43-feature-v1"
    assert loaded.contract_version == "formal-V3-catboost43"
    assert loaded.channel_results[0]["class_probabilities"]
    assert loaded.fused_probabilities is not None
    assert sum(loaded.fused_probabilities) == 1.0


def test_v3_history_uses_separate_database_and_does_not_touch_v2_file(tmp_path: Path) -> None:
    v2_path = tmp_path / "pump_fault_history.sqlite3"
    v2_path.write_bytes(b"old-v2-database-fixture")
    before = v2_path.read_bytes()
    repo = V3HistoryRepository(tmp_path / "pump_fault_history_v3.sqlite3")
    repo.save(
        _diagnosed_result(),
        source_files={"CH3": "a.csv", "CH4": "b.csv"},
        sampling_rate_hz=12_000,
        rpm=1500.0,
    )
    assert v2_path.read_bytes() == before


def test_v3_history_lists_and_deletes_only_selected_record(tmp_path: Path) -> None:
    repo = V3HistoryRepository(tmp_path / "history_v3.sqlite3")
    first = repo.save(_diagnosed_result(), source_files={"CH3": "a", "CH4": "b"}, sampling_rate_hz=12000, rpm=1500)
    second = repo.save(_diagnosed_result(), source_files={"CH3": "c", "CH4": "d"}, sampling_rate_hz=12000, rpm=1500)
    assert len(repo.list_all()) == 2
    assert repo.delete(int(first.id)) is True
    assert repo.get(int(first.id)) is None
    assert repo.get(int(second.id)) == second
