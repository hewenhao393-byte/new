from __future__ import annotations

from pathlib import Path

import pytest

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES
from pump_fault_app.domain.formal_contract import FORMAL_LABEL_ORDER
from pump_fault_app.domain.records import FeatureVector, SampleMetadata
from pump_fault_app.model_registry import SQLiteModelRegistry
from pump_fault_app.model_training import train_candidate_model
from pump_fault_app.sample_repository import SQLiteSampleRepository


def _feature_vector(value: float, *, label: str | None = None) -> FeatureVector:
    return FeatureVector(
        feature_names=FORMAL_FEATURE_NAMES,
        values=tuple(value + index for index in range(len(FORMAL_FEATURE_NAMES))),
        metadata=SampleMetadata(label=label, rpm=1450.0),
    )


def test_sample_repository_stores_snapshots_and_overwrites_annotation(tmp_path: Path) -> None:
    repository = SQLiteSampleRepository(tmp_path / "samples.sqlite3")

    count = repository.store_feature_snapshots(
        history_record_id=7,
        feature_vectors=(_feature_vector(1.0), _feature_vector(2.0)),
    )
    repository.save_annotation(history_record_id=7, true_label="汽蚀")
    annotation = repository.save_annotation(history_record_id=7, true_label="轴承故障")

    assert count == 2
    assert annotation.true_label == "轴承故障"
    rows = repository.list_training_rows()
    assert len(rows) == 2
    assert {row["true_label"] for row in rows} == {"轴承故障"}
    assert all(row["history_record_id"] == 7 for row in rows)


def test_training_rows_exclude_unlabeled_snapshots(tmp_path: Path) -> None:
    repository = SQLiteSampleRepository(tmp_path / "samples.sqlite3")
    repository.store_feature_snapshots(
        history_record_id=3,
        feature_vectors=(_feature_vector(3.0),),
    )

    assert repository.list_training_rows() == []
    assert repository.snapshot_count(history_record_id=3) == 1


def test_sample_repository_rejects_non_formal_feature_order(tmp_path: Path) -> None:
    repository = SQLiteSampleRepository(tmp_path / "samples.sqlite3")
    invalid = FeatureVector(
        feature_names=tuple(reversed(FORMAL_FEATURE_NAMES)),
        values=tuple(float(index) for index in range(len(FORMAL_FEATURE_NAMES))),
        metadata=SampleMetadata(),
    )

    with pytest.raises(ValueError, match="frozen 21-feature order"):
        repository.store_feature_snapshots(history_record_id=1, feature_vectors=(invalid,))


def test_candidate_training_writes_candidate_bundle_and_grouped_metrics(tmp_path: Path) -> None:
    rows = []
    for label_index, label in enumerate(FORMAL_LABEL_ORDER):
        for group_offset in range(5):
            rows.append(
                {
                    "history_record_id": label_index * 100 + group_offset,
                    "window_index": 0,
                    "true_label": label,
                    **{
                        feature: float(label_index * 10 + group_offset + feature_index)
                        for feature_index, feature in enumerate(FORMAL_FEATURE_NAMES)
                    },
                }
            )

    result = train_candidate_model(rows, output_dir=tmp_path)

    assert result.bundle_path.exists()
    assert result.sample_count == 30
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.bundle_path.parent == tmp_path


def test_candidate_training_rejects_missing_class_coverage(tmp_path: Path) -> None:
    row = {
        "history_record_id": 1,
        "window_index": 0,
        "true_label": "正常",
        **{feature: float(index) for index, feature in enumerate(FORMAL_FEATURE_NAMES)},
    }

    with pytest.raises(ValueError, match="six formal labels"):
        train_candidate_model([row], output_dir=tmp_path)


def test_model_registry_approves_candidate_without_replacing_formal_model(tmp_path: Path) -> None:
    registry = SQLiteModelRegistry(tmp_path / "registry.sqlite3")
    candidate = registry.add_candidate(
        version="candidate-20260804-001",
        bundle_path=tmp_path / "candidate.joblib",
        trained_at="2026-08-04 12:00:00",
        sample_count=30,
        accuracy=0.8,
        macro_f1=0.75,
    )

    approved = registry.approve(candidate.version)

    assert approved.status == "已批准（未接入正式推理）"
    assert registry.current_formal_version().version
