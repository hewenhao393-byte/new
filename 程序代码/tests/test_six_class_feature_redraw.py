from __future__ import annotations

import numpy as np
import pandas as pd

from pump_diagnosis.six_class_feature_redraw import (
    CLASS_LAYOUT_ORDER,
    REDRAW_FEATURE_COLUMNS,
    normalize_window_waveform,
    select_representative_windows,
    should_expand_envelope_limit,
)


def test_select_representative_windows_returns_one_window_per_class() -> None:
    frame = pd.DataFrame(
        [
            {
                "label": "正常",
                "device_id": "Motor-2",
                "speed_percent": 100,
                "rpm": 1480,
                "source_file": "a.csv",
                "group_id": "g1",
                "window_id": "w1",
                **{name: 0.0 for name in REDRAW_FEATURE_COLUMNS},
            },
            {
                "label": "正常",
                "device_id": "Motor-2",
                "speed_percent": 100,
                "rpm": 1480,
                "source_file": "a.csv",
                "group_id": "g1",
                "window_id": "w2",
                **{name: 1.0 for name in REDRAW_FEATURE_COLUMNS},
            },
            {
                "label": "转子不平衡",
                "device_id": "Motor-4",
                "speed_percent": 70,
                "rpm": 2070,
                "source_file": "b.csv",
                "group_id": "g2",
                "window_id": "w3",
                **{name: 2.0 for name in REDRAW_FEATURE_COLUMNS},
            },
            {
                "label": "转子不平衡",
                "device_id": "Motor-4",
                "speed_percent": 70,
                "rpm": 2070,
                "source_file": "b.csv",
                "group_id": "g2",
                "window_id": "w4",
                **{name: 3.0 for name in REDRAW_FEATURE_COLUMNS},
            },
        ]
    )

    selected = select_representative_windows(frame)

    assert set(selected) == {"正常", "转子不平衡"}
    assert selected["正常"]["window_id"] == "w1"
    assert selected["转子不平衡"]["window_id"] == "w3"


def test_normalize_window_waveform_scales_to_unit_range() -> None:
    waveform = np.array([0.0, 2.0, -4.0, 1.0], dtype=float)

    normalized = normalize_window_waveform(waveform)

    assert np.isclose(np.max(np.abs(normalized)), 1.0)
    assert np.isclose(normalized[2], -1.0)


def test_should_expand_envelope_limit_only_when_peak_exceeds_300() -> None:
    assert not should_expand_envelope_limit([120.0, 180.0, 250.0])
    assert should_expand_envelope_limit([120.0, 180.0, 305.0])


def test_layout_order_matches_required_2x3_sequence() -> None:
    assert CLASS_LAYOUT_ORDER == [
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    ]
