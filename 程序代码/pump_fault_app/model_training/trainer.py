from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER


@dataclass(frozen=True)
class CandidateTrainingResult:
    version: str
    bundle_path: Path
    trained_at: str
    sample_count: int
    accuracy: float
    macro_f1: float


def train_candidate_model(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    trained_at: datetime | None = None,
) -> CandidateTrainingResult:
    frame = pd.DataFrame(rows)
    _validate_training_frame(frame)
    x = frame.loc[:, FORMAL_FEATURE_NAMES]
    y = frame["true_label"].astype(str)
    groups = frame["history_record_id"]
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predicted = cross_val_predict(pipeline, x, y, groups=groups, cv=splitter)
    pipeline.fit(x, y)
    timestamp = trained_at or datetime.now()
    version = f"candidate-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_path = destination / f"{version}_bp_bundle.joblib"
    joblib.dump(
        {
            "imputer": pipeline.named_steps["imputer"],
            "scaler": pipeline.named_steps["scaler"],
            "model": pipeline.named_steps["model"],
            "features": list(FORMAL_FEATURE_NAMES),
        },
        bundle_path,
    )
    return CandidateTrainingResult(
        version=version,
        bundle_path=bundle_path,
        trained_at=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        sample_count=len(frame),
        accuracy=float(accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, average="macro")),
    )


def _validate_training_frame(frame: pd.DataFrame) -> None:
    required = {"history_record_id", "true_label", *FORMAL_FEATURE_NAMES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training data is missing required columns: {sorted(missing)}")
    labels = set(frame["true_label"].astype(str))
    if labels != set(FORMAL_LABEL_ORDER):
        raise ValueError("training data must cover all six formal labels")
    group_label_counts = frame.groupby("true_label")["history_record_id"].nunique()
    if (group_label_counts < 5).any():
        raise ValueError("each formal label requires at least five confirmed history samples")
