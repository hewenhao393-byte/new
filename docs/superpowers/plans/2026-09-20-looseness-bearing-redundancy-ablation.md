# Looseness/Bearing Diagnosis and Redundancy Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible second-stage experiment that diagnoses looseness/bearing confusion, audits the 19 correlation records, compares the original 43-feature set with one fixed 40-feature set, and selects CatBoost iterations using record-grouped training-only CV.

**Architecture:** Add a separate `ablation_analysis` package so the accepted baseline pipeline remains frozen. The package reads the six source feature tables and six baseline prediction tables, validates their join keys, performs record-balanced diagnostic statistics, defines one deterministic 40-feature contract, selects iterations from staged CV probabilities, trains 12 final models, and writes a new isolated result directory. A final verifier reloads every artifact and recomputes the important metrics before reporting completion.

**Tech Stack:** Python 3.9, pandas, NumPy, SciPy, scikit-learn, CatBoost 1.2.10, matplotlib, pytest.

---

### Task 1: Define immutable experiment contracts and input validation

**Files:**
- Create: `ablation_analysis/__init__.py`
- Create: `ablation_analysis/config.py`
- Create: `ablation_analysis/input_validation.py`
- Create: `tests/ablation_analysis/test_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from ablation_analysis.config import (
    FEATURE_43, FEATURE_40, REMOVED_FEATURES, ITERATION_GRID
)

def test_conservative_feature_contract():
    assert len(FEATURE_43) == 43
    assert REMOVED_FEATURES == [
        "std", "band_energy_3000_5000_ratio", "wp_energy_ratio_7"
    ]
    assert FEATURE_40 == [f for f in FEATURE_43 if f not in REMOVED_FEATURES]
    assert len(FEATURE_40) == 40

def test_iteration_grid_is_fixed_before_training():
    assert ITERATION_GRID == list(range(20, 801, 20))
```

- [ ] **Step 2: Run the tests and confirm the expected import failure**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_contracts.py -q
```

Expected: FAIL because `ablation_analysis` does not exist.

- [ ] **Step 3: Implement the fixed contracts**

```python
# ablation_analysis/config.py
from baseline_analysis.config import FEATURE_COLUMNS, LABEL_ORDER, MODEL_PARAMS

FEATURE_43 = list(FEATURE_COLUMNS)
REMOVED_FEATURES = [
    "std", "band_energy_3000_5000_ratio", "wp_energy_ratio_7"
]
FEATURE_40 = [f for f in FEATURE_43 if f not in REMOVED_FEATURES]
ITERATION_GRID = list(range(20, 801, 20))
MAX_ITERATIONS = 800
CV_SPLITS = 5
JOIN_KEYS = ["record_id", "window_id", "start_sample", "end_sample"]
```

In `input_validation.py`, implement `validate_inputs(feature_tables, prediction_tables)` to assert:

```python
assert set(JOIN_KEYS).issubset(features.columns)
assert not features.duplicated(JOIN_KEYS).any()
assert predictions[JOIN_KEYS].equals(
    features.loc[predictions.index, JOIN_KEYS].reset_index(drop=True)
)
assert predictions["label"].equals(
    features.loc[predictions.index, "label"].reset_index(drop=True)
)
```

Use a key merge with `validate="one_to_one"`; do not rely on row order in the production implementation. Reject unmatched prediction rows, duplicate keys, channel mismatches, split mismatches, label mismatches, and non-finite feature values.

- [ ] **Step 4: Add validation tests with shuffled rows and one deliberate mismatch**

```python
def test_prediction_join_is_key_based_and_rejects_label_mismatch():
    merged = join_predictions_to_features(features, predictions.sample(frac=1))
    assert len(merged) == len(predictions)
    bad = predictions.copy()
    bad.loc[0, "label"] = "wrong"
    with pytest.raises(ValueError, match="label mismatch"):
        join_predictions_to_features(features, bad)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_contracts.py -q
```

Expected: all Task 1 tests PASS.

Commit:

```bash
git add ablation_analysis tests/ablation_analysis
git commit -m "feat: define ablation experiment contracts"
```

### Task 2: Implement record-balanced looseness/bearing diagnostics

**Files:**
- Create: `ablation_analysis/confusion_diagnostics.py`
- Create: `tests/ablation_analysis/test_confusion_diagnostics.py`

- [ ] **Step 1: Write failing tests for four-group assignment**

```python
def test_assigns_only_the_four_target_groups():
    actual = pd.Series(["松动", "松动", "轴承故障", "轴承故障", "正常"])
    predicted = pd.Series(["松动", "轴承故障", "轴承故障", "松动", "正常"])
    assert assign_target_group(actual, predicted).tolist() == [
        "correct_looseness", "looseness_to_bearing",
        "correct_bearing", "bearing_to_looseness", None
    ]
```

- [ ] **Step 2: Write failing tests for Cliff's delta direction and magnitude**

```python
def test_cliffs_delta_has_declared_direction():
    result = cliffs_delta(np.array([1, 1, 2]), np.array([5, 6, 6]))
    assert result["delta"] == -1.0
    assert result["magnitude"] == "large"
```

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_confusion_diagnostics.py -q
```

Expected: FAIL because the diagnostic functions are absent.

- [ ] **Step 4: Implement diagnostics**

Implement the group assignment and record aggregation directly:

```python
TARGET_GROUPS = {
    ("松动", "松动"): "correct_looseness",
    ("松动", "轴承故障"): "looseness_to_bearing",
    ("轴承故障", "轴承故障"): "correct_bearing",
    ("轴承故障", "松动"): "bearing_to_looseness",
}

def assign_target_group(actual, predicted):
    return pd.Series(
        [TARGET_GROUPS.get(pair) for pair in zip(actual, predicted)],
        index=actual.index, dtype="object"
    )

def record_feature_medians(target_windows, features):
    return (target_windows.groupby(["target_group", "record_id"], as_index=False)
            [features].median())

def cliffs_delta(x, y):
    delta = (np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()) / (len(x) * len(y))
    absolute = abs(delta)
    magnitude = "negligible" if absolute < .147 else "small" if absolute < .33 else "medium" if absolute < .474 else "large"
    return {"delta": float(delta), "abs_delta": float(absolute), "magnitude": magnitude}
```

Build `grouped_feature_summary` by iterating over `target_group` and features and emitting count, unique record count, median, Q1, Q3, and IQR. Build `effect_size_table` for the three declared comparisons using the record-median table. Build `targeted_error_concentration` by counting directional error windows per record and computing the top-five share from those counts.

Compute delta as `(number of x>y - number of x<y) / (len(x)*len(y))`. Return `n_records_a`, `n_records_b`, `delta`, `abs_delta`, `magnitude`, and `small_sample`; `small_sample=True` when either group has fewer than five records.

- [ ] **Step 5: Add record-balancing and concentration tests**

Construct one record with 100 windows and another with one window. Assert that `record_feature_medians` returns two observations, not 101. Assert that the top-five concentration is `target_error_windows_in_top5 / all_target_error_windows`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_confusion_diagnostics.py -q
```

Expected: PASS.

Commit:

```bash
git add ablation_analysis/confusion_diagnostics.py tests/ablation_analysis/test_confusion_diagnostics.py
git commit -m "feat: add looseness bearing diagnostics"
```

### Task 3: Consolidate and classify the 19 correlation records

**Files:**
- Create: `ablation_analysis/redundancy.py`
- Create: `tests/ablation_analysis/test_redundancy.py`

- [ ] **Step 1: Write failing tests for pair normalization and consolidation**

```python
def test_reversed_pairs_are_consolidated():
    rows = pd.DataFrame([
        {"feature_a": "rms", "feature_b": "std", "pearson_r": .99, "channel": 3, "split_mode": "record"},
        {"feature_a": "std", "feature_b": "rms", "pearson_r": .98, "channel": 4, "split_mode": "temporal"},
    ])
    out = consolidate_pairs(rows)
    assert len(out) == 1
    assert out.loc[0, "occurrence_count"] == 2
    assert out.loc[0, "min_abs_r"] == .98
```

- [ ] **Step 2: Write failing tests for the decision table**

```python
def test_only_three_predeclared_features_are_removed():
    decisions = redundancy_decisions(FEATURE_43)
    removed = decisions.query("decision == 'remove'")["feature"].tolist()
    assert removed == REMOVED_FEATURES
    assert decisions.query("decision == 'remove'").reason.notna().all()
```

- [ ] **Step 3: Implement consolidation and deterministic decisions**

`consolidate_pairs` must output canonical `feature_a`, `feature_b`, occurrence count, channel list, split list, minimum/maximum absolute Pearson correlation, and correlation signs. `redundancy_decisions` must label:

```python
{
  "std": "near-duplicate of RMS for these vibration windows",
  "band_energy_3000_5000_ratio": "reference component of four-part closed composition",
  "wp_energy_ratio_7": "reference component of eight-part closed composition",
}
```

All other features receive `decision="retain"`; locally correlated pairs receive `review_note="retained because correlation is not stable across all channels"`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_redundancy.py -q
```

Expected: PASS.

Commit:

```bash
git add ablation_analysis/redundancy.py tests/ablation_analysis/test_redundancy.py
git commit -m "feat: audit correlated feature redundancy"
```

### Task 4: Implement leakage-safe staged iteration selection

**Files:**
- Create: `ablation_analysis/iteration_selection.py`
- Create: `tests/ablation_analysis/test_iteration_selection.py`

- [ ] **Step 1: Write a failing fold-isolation test**

```python
def test_cv_folds_have_no_record_overlap():
    folds = make_record_folds(labels, record_ids, n_splits=5, seed=2026)
    for fit_idx, validation_idx in folds:
        assert set(record_ids.iloc[fit_idx]).isdisjoint(record_ids.iloc[validation_idx])
```

- [ ] **Step 2: Write a failing staged-selection test**

```python
def test_selects_best_mean_record_macro_f1_and_smallest_exact_tie():
    scores = pd.DataFrame({
        "iteration": [20, 40, 60],
        "mean_record_macro_f1": [.80, .85, .85],
    })
    assert choose_iteration(scores) == 40
```

- [ ] **Step 3: Run focused tests and confirm failure**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_iteration_selection.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 4: Implement fold generation and staged scoring**

Implement fold generation, aggregation, and deterministic selection as follows:

```python
def make_record_folds(labels, record_ids, n_splits=5, seed=2026):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups=record_ids))

def aggregate_cv_scores(fold_scores):
    return (fold_scores.groupby("iteration", as_index=False)
            .record_macro_f1.agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "mean_record_macro_f1", "std": "std_record_macro_f1"}))

def choose_iteration(summary, atol=1e-12):
    best = summary.mean_record_macro_f1.max()
    tied = summary[np.isclose(summary.mean_record_macro_f1, best, atol=atol, rtol=0)]
    return int(tied.iteration.min())
```

`staged_record_scores` iterates over `model.staged_predict_proba(validation[feature_names])` with a one-based iteration counter, skips counters outside `ITERATION_GRID`, constructs the same probability columns used by `fuse_records`, and emits one record-level Macro-F1 row per checkpoint. `select_iterations` fits one 800-tree model per supplied fold, concatenates these rows, calls `aggregate_cv_scores`, and returns fold rows, summary rows, and `choose_iteration(summary)`.

Train each fold with `iterations=800`. Use CatBoost's staged prediction interface and retain only iterations in `ITERATION_GRID`. At every checkpoint, average class probabilities by `record_id`, then compute six-class record-level Macro-F1 with the fixed `LABEL_ORDER`. Store per-fold scores, mean, standard deviation, selected checkpoint, and fold record IDs.

- [ ] **Step 5: Add a synthetic CatBoost integration test**

Use six classes, at least five records per class, and two windows per record. Assert that all 40 checkpoints are emitted, the chosen iteration belongs to `ITERATION_GRID`, and feature order is preserved.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_iteration_selection.py -q
```

Expected: PASS.

Commit:

```bash
git add ablation_analysis/iteration_selection.py tests/ablation_analysis/test_iteration_selection.py
git commit -m "feat: select CatBoost iterations with record CV"
```

### Task 5: Build the 12-model ablation runner

**Files:**
- Create: `ablation_analysis/modeling.py`
- Create: `ablation_analysis/pipeline.py`
- Create: `run_redundancy_ablation.py`
- Create: `tests/ablation_analysis/test_pipeline.py`

- [ ] **Step 1: Write a failing output-contract test**

```python
def test_run_names_cover_twelve_models():
    runs = build_run_matrix()
    assert len(runs) == 12
    assert set(runs.feature_set) == {"features_43", "features_40"}
    assert set(runs.channel) == {3, 4, 5}
    assert set(runs.split_mode) == {"record", "temporal"}
```

- [ ] **Step 2: Implement final-model training**

`train_selected_model` must accept the selected iteration explicitly and refuse values outside `ITERATION_GRID`. It must save:

```text
model.cbm
metadata.json
internal_cv_fold_scores.csv
internal_cv_iteration_summary.csv
window_predictions.csv
record_predictions.csv
window_metrics.json
record_metrics.json
feature_importance.csv
```

Reuse `baseline_analysis.evaluation.fuse_records` and `evaluate_predictions`. Reload `model.cbm` and assert saved-model probabilities match the in-memory model within `atol=1e-12`.

- [ ] **Step 3: Implement the isolated pipeline**

The CLI is:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python run_redundancy_ablation.py \
  --features '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_两种划分_20260920' \
  --baseline '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_CatBoost基线_20260920' \
  --output '/Users/hewenhao/Documents/特征提取/实验结果/单通道_去冗余消融_迭代选择_20260920'
```

The output directory must not exist. If it exists, raise `FileExistsError`; never overwrite or delete it. Generate one fold manifest per channel/split and reuse that exact manifest for 43 and 40 features.

- [ ] **Step 4: Add a tiny end-to-end test**

Monkeypatch `ITERATION_GRID` to `[2, 4]` and `MAX_ITERATIONS` to `4`; create six-class synthetic data with five records per class. Assert that 12 model directories are created, 43/40 paired runs share the same fold hash, and no test row is present in an internal CV fold manifest.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_pipeline.py -q
```

Expected: PASS.

Commit:

```bash
git add ablation_analysis run_redundancy_ablation.py tests/ablation_analysis/test_pipeline.py
git commit -m "feat: run 43 versus 40 feature ablation"
```

### Task 6: Produce comparison tables, plots, and decision report

**Files:**
- Create: `ablation_analysis/reporting.py`
- Create: `tests/ablation_analysis/test_reporting.py`

- [ ] **Step 1: Write failing decision-rule tests**

```python
def test_recommends_40_only_when_all_guardrails_pass():
    deltas = pd.DataFrame({
        "record_macro_f1_delta": [-.004] * 6,
        "looseness_recall_delta": [-.009] * 6,
        "bearing_recall_delta": [-.009] * 6,
    })
    assert recommend_feature_set(deltas) == "features_40"
    deltas.loc[0, "bearing_recall_delta"] = -.011
    assert recommend_feature_set(deltas) == "features_43"
```

- [ ] **Step 2: Write failing CH5-value tests**

```python
def test_ch5_unique_value_requires_both_splits_same_direction():
    recalls = make_recall_fixture(record_ch5=.95, temporal_ch5=.96,
                                  record_best_other=.94, temporal_best_other=.95)
    assert assess_ch5_unique_value(recalls)["has_unique_value"]
```

- [ ] **Step 3: Implement reporting**

Produce:

```text
diagnostics/four_group_feature_statistics.csv
diagnostics/record_level_cliffs_delta.csv
diagnostics/targeted_error_records.csv
diagnostics/error_concentration.csv
redundancy/consolidated_high_correlation_pairs.csv
redundancy/feature_decisions_43_to_40.csv
iteration_selection/all_iteration_curves.csv
comparison/ablation_metrics.csv
comparison/ablation_deltas.csv
comparison/channel_class_recall.csv
comparison/ch5_unique_value.csv
figures/iteration_curves/*.png
figures/class_recall/*.png
conclusion.md
run_manifest.json
```

The conclusion must have separate sections for training-only CV, independent test-window results, and independent test-record results. It must explicitly state whether 40 features pass both guardrails and whether current evidence justifies adding new features.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/ablation_analysis/test_reporting.py -q
```

Expected: PASS.

Commit:

```bash
git add ablation_analysis/reporting.py tests/ablation_analysis/test_reporting.py
git commit -m "feat: report ablation decisions and channel recalls"
```

### Task 7: Run the real experiment and independently verify every artifact

**Files:**
- Create: `ablation_analysis/verification.py`
- Create: `tests/ablation_analysis/test_verification.py`
- Generate only: `/Users/hewenhao/Documents/特征提取/实验结果/单通道_去冗余消融_迭代选择_20260920/`

- [ ] **Step 1: Add verification tests**

```python
def test_verifier_rejects_corrupted_experiment(tmp_path):
    fixture = build_verified_fixture(tmp_path)
    model = fixture / "models/record/ch3/features_43/model.cbm"
    model.rename(model.with_suffix(".cbm.missing"))
    with pytest.raises(AssertionError, match="12 models"):
        verify_experiment(fixture)

def test_verifier_rejects_fold_leakage(tmp_path):
    fixture = build_verified_fixture(tmp_path)
    fold = pd.read_csv(fixture / "fold_manifests/record_ch3.csv")
    fold.loc[0, "validation_record_id"] = fold.loc[0, "fit_record_id"]
    fold.to_csv(fixture / "fold_manifests/record_ch3.csv", index=False)
    with pytest.raises(AssertionError, match="record overlap"):
        verify_experiment(fixture)

def test_verifier_rejects_invalid_record_probability(tmp_path):
    fixture = build_verified_fixture(tmp_path)
    path = fixture / "models/record/ch3/features_43/record_predictions.csv"
    rows = pd.read_csv(path)
    rows.loc[0, LABEL_ORDER] = 0.5
    rows.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="probability sum"):
        verify_experiment(fixture)
```

Add two corresponding fixtures for an iteration value of 810 and a comparison Macro-F1 changed by 0.01; assert messages contain `iteration grid` and `metric mismatch` respectively.

- [ ] **Step 2: Run the full automated test suite**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ch43_ablation_pycache \
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest \
  tests/baseline_analysis tests/ablation_analysis -q
```

Expected: all tests PASS; warnings may mention pytest cache permissions but no test may fail.

- [ ] **Step 3: Run one real smoke experiment without touching the final output path**

Run the CH3 record 40-feature configuration into a new `/private/tmp/ch43_ablation_smoke_<timestamp>` directory with one CV fold or a bounded debug flag. Confirm staged checkpoints, model reload equality, window predictions, record predictions, and metrics are produced. Never reuse or delete an earlier smoke directory.

- [ ] **Step 4: Run all real diagnostics and 12 models**

Run the CLI command from Task 5 exactly once against the non-existent final output directory. Track progress by completed model directories. Do not use the independent test metrics to alter `FEATURE_40`, folds, the iteration grid, or selected iterations.

- [ ] **Step 5: Execute independent verification**

`verify_experiment(output_root)` must assert:

```python
assert model_count == 12
assert all(selected_iteration in ITERATION_GRID)
assert all(fit_records.isdisjoint(validation_records))
assert all(record_predictions.valid_window_count == record_predictions.window_count)
assert all(np.isclose(probability_sums, 1.0, atol=1e-8))
assert all(recomputed_metrics_match_saved_metrics)
assert paired_43_40_fold_hashes_match
assert diagnostic_join_unmatched_rows == 0
```

Write `verification_report.csv` and `verification_summary.md` to the final output directory.

- [ ] **Step 6: Visually inspect representative figures and review conclusions**

Open at least one iteration curve, one class-recall chart, and the conclusion report. Confirm Chinese labels render, curves contain 40 checkpoints, the selected iteration marker matches the CSV, and every recommendation follows the declared guardrails.

- [ ] **Step 7: Commit implementation and verification code**

Do not commit generated models or large experiment outputs.

```bash
git add ablation_analysis tests/ablation_analysis run_redundancy_ablation.py
git commit -m "test: verify redundancy ablation experiment"
```

## Final acceptance checklist

- The existing six feature tables and baseline directory are unchanged.
- The new result directory contains four-group diagnostics for all six baseline runs.
- The 19 correlation records are consolidated and physically classified; exactly three predeclared features form the 40-feature set.
- The 43/40 paired runs share fold manifests but select iterations independently.
- All 12 selected iterations come only from training-side record-level Macro-F1.
- Internal CV, test-window, and test-record results are presented separately.
- Six-class Recall comparisons explicitly assess whether CH5 has consistent unique value.
- The report recommends 40 features only if every declared performance guardrail passes.
- The report recommends new features only if the declared confusion/effect-size evidence supports expansion.
- No file is deleted and no existing result is overwritten.
