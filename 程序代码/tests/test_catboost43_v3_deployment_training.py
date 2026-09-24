from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER
from tools.train_catboost43_v3_deployment import (
    DEPLOYMENT_SOURCES,
    FIXED_PARAMS,
    SOURCE_ROOT,
    validate_feature_tables,
)


def test_training_script_can_be_executed_directly() -> None:
    script = Path(__file__).resolve().parents[1] / "tools/train_catboost43_v3_deployment.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-root" in completed.stdout


def test_training_uses_only_file_split_tables() -> None:
    assert DEPLOYMENT_SOURCES == {
        "CH3": SOURCE_ROOT / "file_split/features_ch3.csv",
        "CH4": SOURCE_ROOT / "file_split/features_ch4.csv",
        "CH5": SOURCE_ROOT / "file_split/features_ch5.csv",
    }
    assert all("temporal_split" not in str(path) for path in DEPLOYMENT_SOURCES.values())


def test_fixed_params_are_p1_at_500() -> None:
    assert FIXED_PARAMS == {
        "loss_function": "MultiClass",
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 100,
        "random_strength": 5,
        "rsm": 0.7,
        "auto_class_weights": "SqrtBalanced",
        "random_seed": 2026,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": 4,
    }


def _frame(channel: int) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(FORMAL_LABEL_ORDER):
        row = {
            "record_id": f"record-{index}",
            "group_id": f"record-{index}",
            "window_id": 0,
            "channel": channel,
            "label": label,
        }
        row.update({name: float(index + feature_index) for feature_index, name in enumerate(FORMAL_FEATURE_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_validation_accepts_aligned_three_channel_tables() -> None:
    tables = {f"CH{channel}": _frame(channel) for channel in (3, 4, 5)}
    validate_feature_tables(tables)


def test_validation_rejects_sibling_key_misalignment() -> None:
    tables = {f"CH{channel}": _frame(channel) for channel in (3, 4, 5)}
    tables["CH5"].loc[0, "record_id"] = "different-record"
    with pytest.raises(ValueError, match="sibling channel keys"):
        validate_feature_tables(tables)


def test_validation_rejects_duplicate_windows() -> None:
    tables = {f"CH{channel}": _frame(channel) for channel in (3, 4, 5)}
    tables["CH3"] = pd.concat([tables["CH3"], tables["CH3"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate window keys"):
        validate_feature_tables(tables)
