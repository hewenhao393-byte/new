import numpy as np
import pandas as pd
import pytest

from feature_pipeline.config import CONFIG, FEATURE_NAMES
from feature_pipeline.metadata import make_record_id, merge_measurement_metadata
from feature_pipeline.preprocessing import assess_quality, preprocess_record
from feature_pipeline.splits import make_record_split, temporal_bounds
from feature_pipeline.windowing import iter_temporal_windows


def test_formal_contract_is_fixed():
    assert (CONFIG.original_fs, CONFIG.target_fs) == (20_000, 12_000)
    assert (CONFIG.window_size, CONFIG.step_size) == (4_800, 2_400)
    assert (CONFIG.temporal_train_samples, CONFIG.temporal_guard_samples, CONFIG.temporal_test_samples) == (96_000, 12_000, 36_000)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 43


def test_record_identity_and_unmatched_severity():
    assert make_record_id("Motor-2", 100, "正常状态1", "0") == "Motor-2::100::正常状态1::0"
    source = pd.DataFrame([{"group_id": "g", "device_id": "Motor-2", "speed_percent": 100, "raw_fault": "正常状态1"}])
    merged, audit = merge_measurement_metadata(source, [])
    assert pd.isna(merged.loc[0, "severity"])
    assert audit.loc[0, "status"] == "unmatched"


def test_preprocessing_and_impulse_policy():
    t = np.arange(240_000) / 20_000
    y = preprocess_record(np.sin(2 * np.pi * 100 * t) + 0.5)
    assert y.shape == (144_000,)
    assert np.isfinite(y).all()
    x = np.zeros(240_000); x[1000] = 100
    assert "high_kurtosis" not in assess_quality(x).rejection_reasons


def test_fixed_temporal_bounds_and_block_safe_windows():
    tail = temporal_bounds("a", 144_000, force_direction="test_tail")
    assert tail["train"] == (0, 96_000)
    assert tail["guard"] == (96_000, 108_000)
    assert tail["test"] == (108_000, 144_000)
    windows = list(iter_temporal_windows(144_000, tail))
    assert all(not (w.start < 108_000 and w.end > 96_000) for w in windows)
    assert all(w.split != "train" or (w.start // 19_200) == ((w.end - 1) // 19_200) for w in windows)


def test_record_split_is_unique_and_stable():
    rows = []
    for label in ("正常", "松动"):
        for i in range(10):
            rows.append({"record_id": f"{label}-{i}", "label": label, "device_id": "M", "speed_percent": 100, "raw_fault": label})
    records = pd.DataFrame(rows)
    first = make_record_split(records)
    second = make_record_split(records)
    pd.testing.assert_frame_equal(first, second)
    assert first.record_id.is_unique
    assert set(first.split) == {"train_dev", "test"}


def test_temporal_requires_exact_12_seconds():
    with pytest.raises(ValueError, match="144000"):
        temporal_bounds("bad", 143_999)
