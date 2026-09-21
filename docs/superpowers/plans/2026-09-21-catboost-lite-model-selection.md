# CatBoost Lite Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one unified CatBoost parameter set from P1/P2/P5 using shared three-fold record CV and seven checkpoints, then train and report six final 43-feature models.

**Architecture:** Add a separate `formal_model_selection` package that reuses accepted feature contracts and evaluation utilities but does not depend on the previous redundancy experiment. Preflight all source tables, build one shared record fold, run 27 bounded CV fits, apply a deterministic selection rule, train six final models, generate thesis-ready tables/figures, and atomically publish a new output directory.

**Tech Stack:** Python 3.9, pandas, NumPy, scikit-learn, CatBoost 1.2.10, matplotlib, pytest.

---

### Task 1: Contracts, folds, and selection rule

Create `formal_model_selection/config.py`, `selection.py`, and `tests/formal_model_selection/test_selection.py`. Tests must prove exact candidates P1/P2/P5, checkpoints `[50,100,150,200,300,400,500]`, one shared deterministic 3-fold record manifest, no record overlap, per-channel mean/std, equal-channel Score/Stability, the 0.002 candidate band, and tie-break order depth→iterations→L2→Stability→ID.

Run:

```bash
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/formal_model_selection/test_selection.py -q
```

Commit: `feat: define lite CatBoost selection contracts`.

### Task 2: Staged three-channel CV

Create `formal_model_selection/cv.py` and tests. Train one 500-tree CatBoost model per candidate/channel/fold, consume one staged prediction stream, retain only seven checkpoints, fuse validation probabilities by record, and save fold/channel Macro-F1. Reject metadata features, test rows, wrong folds, missing checkpoints, invalid probabilities, and unequal fold hashes between channels. Add a small real CatBoost integration test.

Commit: `feat: evaluate unified CatBoost candidates`.

### Task 3: Final six models and error outputs

Create `formal_model_selection/modeling.py` and tests. Lock the selected parameter/checkpoint, fit CH3/4/5 for record and temporal splits, reload each model at `atol=1e-12`, and save required window/record predictions, two-level metrics/confusion matrices, feature importance, targeted directional errors, misclassified records, and error aggregation by motor/rpm/condition/state/severity.

Commit: `feat: train final unified CatBoost models`.

### Task 4: Atomic pipeline and thesis report

Create `formal_model_selection/pipeline.py`, `reporting.py`, `run_formal_model_selection.py`, and tests. Preflight and hash six source tables before writing; use a unique sibling staging directory; generate candidate/checkpoint tables, selected-parameter rationale, six-class channel/split comparisons, Chinese confusion matrices, feature-importance summaries, error-case tables, and a conclusion with separate CV/window/record sections. Rehash inputs before atomic rename. Never auto-run other candidates or five-fold CV.

Commit: `feat: report formal CatBoost model selection`.

### Task 5: Real run and independent verification

Create `formal_model_selection/verification.py` and tests. Verify source hashes, exact shared folds, 27 CV fit combinations and seven checkpoints, deterministic selected combination, six model tree counts/features/probabilities, recomputed metrics/confusion matrices/importance/errors, and readable figures. Run one bounded smoke in a new `/private/tmp` directory, then execute the production CLI once against the non-existent final path. Save `verification_report.csv` and `verification_summary.md`; do not commit generated models.

Run full scoped tests and visually inspect one parameter curve, one confusion matrix, and the conclusion.

Commit: `test: verify formal CatBoost selection`.
