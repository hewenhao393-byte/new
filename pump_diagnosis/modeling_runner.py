from __future__ import annotations

import json

import pandas as pd

from config import ModelingConfig
from pump_diagnosis.features import FEATURE_COLUMNS
from pump_diagnosis.model_training import train_and_evaluate_models
from pump_diagnosis.modeling import (
    audit_leakage,
    drop_nonfinite_feature_rows,
    evaluate_top_k_grouped,
    fit_training_feature_filter,
)


def run_full_modeling(config: ModelingConfig) -> dict[str, object]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.feature_root / "train_features_raw.csv")
    test = pd.read_csv(config.feature_root / "test_features_raw.csv")
    leakage_report = audit_leakage(train, test)
    train, test, nonfinite_report = drop_nonfinite_feature_rows(train, test, FEATURE_COLUMNS)
    filter_result = fit_training_feature_filter(train[FEATURE_COLUMNS], FEATURE_COLUMNS, config)
    top_k_result = evaluate_top_k_grouped(train, filter_result.kept_features, config)
    metrics = train_and_evaluate_models(train, test, top_k_result.selected_features, config)

    pd.DataFrame(
        [{"feature": feature, "reason": reason} for feature, reason in filter_result.removed_reasons.items()]
    ).to_csv(config.output_root / "removed_features.csv", index=False)
    top_k_result.summary.to_csv(config.output_root / "top_k_cv_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": top_k_result.selected_features,
            "importance": list(reversed(range(1, len(top_k_result.selected_features) + 1))),
        }
    ).to_csv(config.output_root / "feature_importance.csv", index=False)
    pd.Series(top_k_result.selected_features).to_json(
        config.output_root / "selected_features.json",
        force_ascii=False,
        indent=2,
    )
    metrics.to_csv(config.output_root / "model_metrics.csv", index=False)

    summary = {
        "leakage_check": leakage_report,
        "nonfinite_report": nonfinite_report,
        "selected_k": top_k_result.selected_k,
        "selected_features": top_k_result.selected_features,
    }
    (config.output_root / "run_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
