from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pump_diagnosis.six_class_feature_redraw import (
    CLASS_LAYOUT_ORDER,
    REDRAW_FEATURE_COLUMNS,
    _create_page,
    RedrawConfig,
    normalize_window_waveform,
    select_representative_windows,
    should_expand_envelope_limit,
    write_representative_window_manifest,
)


def _build_full_coverage_frame() -> pd.DataFrame:
    rows = []
    for label, device_id, speed_percent, rpm, source_file, group_id, window_prefix in [
        ("正常", "Motor-2", 100, 1480, "a.csv", "g1", "n"),
        ("转子不平衡", "Motor-4", 70, 2070, "b.csv", "g2", "u"),
        ("联轴器不对中", "Motor-4", 70, 2070, "c.csv", "g3", "m"),
        ("松动", "Motor-2", 75, 1110, "d.csv", "g4", "l"),
        ("轴承故障", "Motor-2", 50, 740, "e.csv", "g5", "b"),
        ("汽蚀", "Motor-4", 70, 2070, "f.csv", "g6", "c"),
    ]:
        for suffix, value in [("1", 0.0), ("2", 2.0), ("3", 10.0)]:
            rows.append(
                {
                    "label": label,
                    "device_id": device_id,
                    "speed_percent": speed_percent,
                    "rpm": rpm,
                    "source_file": source_file,
                    "group_id": group_id,
                    "window_id": f"{window_prefix}{suffix}",
                    **{name: value for name in REDRAW_FEATURE_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def test_select_representative_windows_returns_one_window_per_class() -> None:
    frame = _build_full_coverage_frame()

    selected = select_representative_windows(frame)

    assert list(selected) == CLASS_LAYOUT_ORDER
    assert selected["正常"]["window_id"] == "n2"
    assert selected["转子不平衡"]["window_id"] == "u2"
    assert selected["联轴器不对中"]["window_id"] == "m2"
    assert selected["松动"]["window_id"] == "l2"
    assert selected["轴承故障"]["window_id"] == "b2"
    assert selected["汽蚀"]["window_id"] == "c2"


def test_select_representative_windows_rejects_missing_feature_columns() -> None:
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
                **{name: 0.0 for name in REDRAW_FEATURE_COLUMNS[:-1]},
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required redraw feature columns"):
        select_representative_windows(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_select_representative_windows_rejects_non_finite_values(bad_value: float) -> None:
    frame = _build_full_coverage_frame()
    frame.loc[0, REDRAW_FEATURE_COLUMNS[0]] = bad_value

    with pytest.raises(ValueError, match="non-finite redraw feature values"):
        select_representative_windows(frame)


def test_select_representative_windows_returns_empty_mapping_for_empty_frame() -> None:
    frame = pd.DataFrame(columns=["label", *REDRAW_FEATURE_COLUMNS])

    selected = select_representative_windows(frame)

    assert selected == {}


def test_select_representative_windows_rejects_incomplete_class_coverage() -> None:
    frame = _build_full_coverage_frame()
    frame = frame[frame["label"] != "汽蚀"].reset_index(drop=True)

    with pytest.raises(ValueError, match="missing required redraw labels"):
        select_representative_windows(frame)


def test_normalize_window_waveform_scales_to_unit_range() -> None:
    waveform = np.array([0.0, 2.0, -4.0, 1.0], dtype=float)

    normalized = normalize_window_waveform(waveform)

    assert np.isclose(np.max(np.abs(normalized)), 1.0)
    assert np.isclose(normalized[2], -1.0)


def test_should_expand_envelope_limit_only_when_peak_exceeds_300() -> None:
    assert not should_expand_envelope_limit([120.0, 180.0, 250.0])
    assert not should_expand_envelope_limit([300.0])
    assert not should_expand_envelope_limit([120.0, 180.0, 305.0])
    assert should_expand_envelope_limit([120.0, 180.0, 305.0, 320.0, 340.0, 360.0])


def test_normalize_window_waveform_keeps_zero_waveform_at_zero() -> None:
    waveform = np.zeros(6, dtype=float)

    normalized = normalize_window_waveform(waveform)

    assert np.array_equal(normalized, waveform)
    assert normalized.dtype == float


def test_normalize_window_waveform_handles_empty_waveform() -> None:
    waveform = np.array([], dtype=float)

    normalized = normalize_window_waveform(waveform)

    assert normalized.size == 0


def test_layout_order_matches_required_2x3_sequence() -> None:
    assert CLASS_LAYOUT_ORDER == [
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    ]


def test_write_representative_window_manifest_preserves_required_columns(tmp_path: Path) -> None:
    rows = [
        {
            "label": "正常",
            "device_id": "Motor-2",
            "speed_percent": 100,
            "rpm": 1480.0,
            "source_file": "x.csv",
            "group_id": "g1",
            "window_id": "w1",
        }
    ]
    output = tmp_path / "representative_windows.csv"

    write_representative_window_manifest(rows, output)

    frame = pd.read_csv(output)
    assert frame.columns.tolist()[:7] == [
        "label",
        "device_id",
        "speed_percent",
        "rpm",
        "source_file",
        "group_id",
        "window_id",
    ]


def test_redraw_config_uses_thesis_window_defaults(tmp_path: Path) -> None:
    config = RedrawConfig(output_root=tmp_path)

    assert config.output_root == tmp_path
    assert config.processed_fs == 12_000
    assert config.window_size == 2400
    assert config.window_step == 1200


def test_create_page_matches_required_three_by_two_layout_and_a4ish_size() -> None:
    fig, axes = _create_page("示例图")

    size_inches = fig.get_size_inches()
    positions = [ax.get_position().bounds for ax in axes]

    assert len(axes) == 6
    assert np.isclose(size_inches[0], 16.0 / 2.54, atol=0.1)
    assert (18.0 / 2.54) <= size_inches[1] <= (20.0 / 2.54)
    assert np.isclose(positions[0][1], positions[1][1], atol=1e-3)
    assert np.isclose(positions[2][1], positions[3][1], atol=1e-3)
    assert np.isclose(positions[4][1], positions[5][1], atol=1e-3)
    assert positions[0][1] > positions[2][1] > positions[4][1]
    assert positions[0][0] < positions[1][0]
    assert positions[2][0] < positions[3][0]
    assert positions[4][0] < positions[5][0]

    fig.clf()


def test_current_redraw_envelope_export_uses_300hz_limit() -> None:
    output = RedrawConfig().output_root / "fig4_6_envelope_spectrum.csv"
    frame = pd.read_csv(output)

    assert output.exists()
    assert set(frame["envelope_limit_hz"].astype(float)) == {300.0}
