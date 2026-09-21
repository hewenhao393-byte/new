# Single-Channel CatBoost Baseline Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revalidate six 43-feature tables, analyze feature quality and training-only correlations, train six fixed-parameter single-channel CatBoost baselines, and deliver independently verified window-, record-, and CV-level results.

**Architecture:** Add a separate `baseline_analysis` package that reads the immutable CSV tables, fails closed on acceptance errors, emits source-backed analysis tables, and trains one model per channel/split with a shared fixed configuration. Each stage writes inspectable CSV/JSON artifacts into a fresh result directory; test data is opened for model evaluation only after training-only work completes.

**Tech Stack:** Python, pandas, NumPy, SciPy, scikit-learn, CatBoost, Matplotlib, pytest.

---

### Task 1: Acceptance engine

**Files:** Create `baseline_analysis/config.py`, `baseline_analysis/acceptance.py`; test `tests/baseline_analysis/test_acceptance.py`.

- [ ] Write failing tests for exact 43-column order, non-finite rejection, channel-key/label/RPM alignment, record leakage, 12000-point guard, guard-crossing windows, train-block crossing, wavelet sum tolerance `1e-8`, and band sum tolerance `1e-6`.
- [ ] Run `pytest -q tests/baseline_analysis/test_acceptance.py` and confirm missing-module failures.
- [ ] Implement chunk-safe six-table reads and an acceptance result containing check name, measured value, threshold, pass flag, and detail path.
- [ ] Re-run the focused tests and commit `feat: validate six final feature tables`.

### Task 2: Feature-quality analysis

**Files:** Create `baseline_analysis/quality.py`; test `tests/baseline_analysis/test_quality.py`.

- [ ] Write failing tests for mean/std/median/Q1/Q3/IQR, pooled-SD effect size, robust median/IQR difference, and record-first variability aggregation.
- [ ] Implement grouped summaries for label, motor, rpm, condition, state, and available severity, keeping split subsets separate.
- [ ] Emit focused normal-vs-misalignment and looseness-vs-bearing comparisons without using test summaries for model choices.
- [ ] Verify controlled calculations and commit `feat: analyze 43-feature quality`.

### Task 3: Training-only correlations

**Files:** Create `baseline_analysis/correlation.py`; test `tests/baseline_analysis/test_correlation.py`.

- [ ] Write failing tests proving only `train_dev`/`train` rows are selected, Pearson `|r|>=0.95` controls the formal list, Spearman is supplemental, and protected order/harmonic features are marked manual-review.
- [ ] Implement six Pearson and six Spearman matrices, pair lists, and deterministic deletion suggestions without deleting columns.
- [ ] Verify no test row participates and commit `feat: report training-only feature correlations`.

### Task 4: Metrics, record fusion, and error attribution

**Files:** Create `baseline_analysis/evaluation.py`; test `tests/baseline_analysis/test_evaluation.py`.

- [ ] Write failing tests for probability validation, record probability means, `window_count`, `valid_window_count`, `mean_max_class_probability`, Accuracy/Macro-F1/Weighted-F1/class metrics, normal false-positive rate, confusion matrices, and targeted error concentration.
- [ ] Implement window and record prediction tables with metadata preservation and six-class fixed ordering.
- [ ] Verify formulas independently and commit `feat: evaluate window and record predictions`.

### Task 5: Fixed CatBoost baseline and internal CV

**Files:** Create `baseline_analysis/modeling.py`; test `tests/baseline_analysis/test_modeling.py`.

- [ ] Write failing tests for the exact fixed baseline contract, no scaler, record groups, globally unique temporal block groups, fold-count fallback, and saved/reloaded probability equality.
- [ ] Implement fixed 540-iteration CatBoost training, StratifiedGroupKFold stability reporting, final all-training fit, one-time test prediction, and PredictionValuesChange importance.
- [ ] Label 540 as a fixed baseline rather than an optimum in every metadata/report output.
- [ ] Run one controlled end-to-end model and commit `feat: train fixed single-channel CatBoost baseline`.

### Task 6: Six-experiment orchestration

**Files:** Create `baseline_analysis/pipeline.py`, `baseline_analysis/main.py`; test `tests/baseline_analysis/test_pipeline.py`.

- [ ] Write a failing synthetic two-mode orchestration test that requires six independent result directories and prohibits fused inputs.
- [ ] Implement immutable input/config hashes, fresh-output refusal, per-run model/prediction/metrics/confusion/error/importance artifacts, and optional bounded SHAP sampling.
- [ ] Run the complete package tests; then run real-table acceptance and one real baseline before the remaining five.
- [ ] Commit `feat: orchestrate six CatBoost baseline experiments`.

### Task 7: Comparison report and final validation

**Files:** Create `baseline_analysis/report.py`; outputs under `实验结果/单通道43维_CatBoost基线_20260920/`.

- [ ] Generate the six-row comparison table, six confusion-matrix CSV/PNG pairs, feature importance/top-20 tables, error-record lists, acceptance report, feature statistics, correlation artifacts, and `conclusion_report.md`.
- [ ] Keep internal CV, independent test-window, and independent test-record sections separate.
- [ ] Reopen all CSV/JSON/CBM/PNG artifacts, reload all models, reconcile comparison values to per-run metrics, and rerun the full test suite.
- [ ] Record supported conclusions, especially temporal-vs-record generalization, without cross-device claims.
- [ ] Commit source/tests/docs only; do not commit generated large artifacts unless explicitly requested.
