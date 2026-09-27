# Three-Channel Fusion Weight Study Implementation Plan

> **For Codex:** Execute this plan task-by-task with test-driven development. Do not modify `pump_fault_app`, formal deployment models, or the existing strict `parent_group` result directory. Do not delete files without the user's explicit approval.

**Goal:** Compare equal, global Macro-F1-weighted, and classwise F1-weighted probability fusion for unchanged CH3/CH4/CH5 CatBoost43 P1@500 models, select only from train-side grouped CV, then run the frozen strategy once on the strict `parent_group` holdout.

**Architecture:** Add a standalone `fusion_weight_study` experimental package and one CLI under `程序代码/`. It consumes the existing three feature tables, strict split file, and CV fold definition as immutable inputs. It regenerates channel probabilities with fixed model settings, constructs leakage-safe cross-fitted fusion results, writes a new atomic experiment directory, and never imports or changes formal inference behavior.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, CatBoost, matplotlib/seaborn, pytest, JSON/CSV/Markdown.

---

## Fixed experiment contract

- Branch/worktree: `codex/channel-fusion-weight-study` in `/Users/hewenhao/.codex/worktrees/channel-fusion-weight-study/特征提取`.
- Immutable source feature tables:
  - `/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_两种划分_20260920/file_split/features_ch3.csv`
  - corresponding `features_ch4.csv` and `features_ch5.csv`.
- Immutable strict split/fold contract:
  - `/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_parent_group严格泛化_20260921/parent_group_split.csv`
  - `/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_parent_group严格泛化_20260921/cv_fold_details.csv`
  - its `run_manifest.json` supplies label order, 43-feature order, P1@500 parameters, source hashes, `cv_splits=3`, and `parent_identity`.
- Six-class order: `正常`, `转子不平衡`, `联轴器不对中`, `松动`, `轴承故障`, `汽蚀`.
- Window alignment key: `(record_id, window_id, start_sample, end_sample)`; labels and split metadata must agree across channels. Any duplicate, missing peer, non-finite feature, hash mismatch, or train/strict overlap is a hard failure.
- Record probability: mean of a channel's window probabilities within `record_id`, followed by channel fusion. This avoids giving records with more windows extra weight at record-level evaluation.
- Equal fusion: arithmetic mean over available channel probability vectors.
- Global fusion: `w_c = MacroF1_c / sum_c MacroF1_c`.
- Classwise fusion: `w_c,k = F1_c,k / sum_c F1_c,k`; compute class score `s_k = sum_c(w_c,k p_c,k) / sum_c(w_c,k)`, then normalize all six `s_k` to sum to one. If a denominator is zero, use equal available-channel weight for that class.
- For every method, a 1-, 2-, or 3-channel subset is legal and its available weights are renormalized. No hard-label voting and no signal-quality weighting.
- Train-side comparison is leakage-safe: for outer fold `j`, derive global/classwise weights only from the other folds' OOF predictions, then score fold `j`. Pool all cross-fitted outer predictions for the selection table.
- Strategy selection uses train-side record-level Macro-F1 as the primary metric, window-level Macro-F1 as secondary, then simplicity within a predeclared tolerance of `0.005`: if weighted improvement over equal is less than 0.005, keep equal; between weighted methods within 0.005, prefer global. Strict results are reported but cannot change the selected strategy.
- Final deployment-candidate weights are recomputed from all train-dev OOF predictions only after selection. The chosen method and weights are frozen before strict evaluation.
- Output default: `/Users/hewenhao/.codex/worktrees/channel-fusion-weight-study/特征提取/实验结果/三通道融合权重实验_20260927/`. Abort if it exists; write to a sibling staging directory and rename atomically only after validation.

## Task 1: Lock contracts and immutable inputs

**Files:**
- Create: `程序代码/fusion_weight_study/__init__.py`
- Create: `程序代码/fusion_weight_study/contracts.py`
- Create: `程序代码/tests/test_fusion_weight_study_contracts.py`

- [ ] Write failing tests that assert the six labels, three channel IDs, exactly 43 ordered features, exact P1@500 parameters, 3 CV folds, primary selection metric, and tolerance.
- [ ] Run `cd '程序代码' && pytest -q tests/test_fusion_weight_study_contracts.py`; expect import failure.
- [ ] Implement frozen dataclasses/constants and a manifest loader that checks the existing manifest against them instead of silently accepting drift.
- [ ] Re-run the test; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study 程序代码/tests/test_fusion_weight_study_contracts.py && git commit -m "test: lock fusion study contract"`.

## Task 2: Validate and align the three channel tables

**Files:**
- Create: `程序代码/fusion_weight_study/data.py`
- Create: `程序代码/tests/test_fusion_weight_study_data.py`

- [ ] Build small synthetic CH3/CH4/CH5 frames and write failing tests for exact feature order, unique alignment keys, channel identity, identical labels/metadata, strict/train separation, fold coverage, and SHA-256 recording.
- [ ] Add explicit negative tests for a missing window, duplicate key, mismatched label, non-finite feature, unknown split, and any parent present in both train-dev and strict sets.
- [ ] Run the focused test and confirm failure.
- [ ] Implement loaders that preserve source rows, join only by the stable window key, derive `parent_group_id` using the manifest rule, and return separate train-dev/strict aligned objects.
- [ ] Validate that every train-dev parent has exactly one validation role in the reused 3-fold file and no strict parent appears in any fit/validation fold.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/data.py 程序代码/tests/test_fusion_weight_study_data.py && git commit -m "feat: validate fusion study inputs"`.

## Task 3: Implement probability fusion for all channel subsets

**Files:**
- Create: `程序代码/fusion_weight_study/fusion.py`
- Create: `程序代码/tests/test_fusion_weight_study_fusion.py`

- [ ] Write parameterized failing tests over all seven non-empty channel subsets for equal, global, and classwise fusion.
- [ ] Assert outputs have six finite non-negative probabilities summing to one, unavailable channels are ignored, and remaining weights are renormalized.
- [ ] Test F1-derived weight calculation, zero-score fallback, classwise post-normalization, deterministic ties, and rejection of hard labels/wrong probability shape.
- [ ] Implement `derive_global_weights`, `derive_classwise_weights`, and `fuse_probabilities` as pure functions.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/fusion.py 程序代码/tests/test_fusion_weight_study_fusion.py && git commit -m "feat: add channel probability fusion"`.

## Task 4: Reproduce fixed single-channel OOF and strict probabilities

**Files:**
- Create: `程序代码/fusion_weight_study/modeling.py`
- Create: `程序代码/tests/test_fusion_weight_study_modeling.py`

- [ ] Write failing tests around a model factory spy: exact CatBoost parameters, six-class probability order, no scaler, fit rows restricted to the fold's `fit` parents, and validation rows restricted to its `validation` parents.
- [ ] Implement fold training from the reused fold map. Train three fixed channel models per fold and emit window OOF probabilities plus fold/model metadata and hashes.
- [ ] After train-side work is complete, train one fixed model per channel on all train-dev rows and predict strict rows. Do not load or overwrite formal deployment models or the old strict models.
- [ ] Add coverage assertions: each train-dev aligned window receives one and only one OOF prediction per channel; each strict window receives one prediction per channel.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/modeling.py 程序代码/tests/test_fusion_weight_study_modeling.py && git commit -m "feat: generate fixed channel probabilities"`.

## Task 5: Build leakage-safe cross-fitted weighting and selection

**Files:**
- Create: `程序代码/fusion_weight_study/experiment.py`
- Create: `程序代码/tests/test_fusion_weight_study_experiment.py`

- [ ] Write failing tests with deliberately distinctive fold scores proving outer fold `j` cannot influence its own global or classwise weights.
- [ ] For each outer fold, derive channel metrics and weights from OOF predictions belonging to the other two folds, fuse outer-fold predictions, and retain the exact weight provenance.
- [ ] Pool cross-fitted predictions and calculate the train-side selection table. Apply the declared record-Macro-F1/secondary/simplicity rule without reading strict metrics.
- [ ] Recompute final global/classwise weights from all train-dev OOF predictions and freeze only the chosen method plus its weights.
- [ ] Run the frozen method against strict probabilities exactly once; ensure a test fails if strict results are passed into the selector.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/experiment.py 程序代码/tests/test_fusion_weight_study_experiment.py && git commit -m "feat: add leakage-safe fusion experiment"`.

## Task 6: Metrics, disagreements, and overdependence diagnostics

**Files:**
- Create: `程序代码/fusion_weight_study/metrics.py`
- Create: `程序代码/tests/test_fusion_weight_study_metrics.py`

- [ ] Write failing tests for window- and record-level Macro-F1, class recall/F1, fixed-label confusion matrices, and probability aggregation.
- [ ] Define disagreement as the number/share of samples where available channel argmax labels are not all identical; report by split, level, true class, and channel-label pattern.
- [ ] Define overdependence diagnostics without tuning thresholds from strict data: largest global weight, per-class largest weight, leave-one-channel-out Macro-F1 deltas, and prediction flip rates. Flag train-side weights above `0.60` or leave-one-channel-out drops above `0.05` as descriptive risks, not model-selection rules.
- [ ] Produce metrics for every method and all seven channel subsets on train-side cross-fitted predictions; on strict, produce the three requested methods with frozen final weights plus leave-one-out diagnostics.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/metrics.py 程序代码/tests/test_fusion_weight_study_metrics.py && git commit -m "feat: add fusion metrics and diagnostics"`.

## Task 7: Write complete, auditable artifacts atomically

**Files:**
- Create: `程序代码/fusion_weight_study/reporting.py`
- Create: `程序代码/tests/test_fusion_weight_study_reporting.py`

- [ ] Write a failing artifact-contract test requiring:
  - `fusion_summary.csv`
  - `channel_weights.json`
  - `classwise_weights.json`
  - `train_side_cv/fold_metrics.csv`, `train_side_cv/per_class_metrics.csv`, `train_side_cv/disagreement.csv`, `train_side_cv/predictions.csv`
  - `strict_parent_group/metrics.csv`, `strict_parent_group/per_class_metrics.csv`, `strict_parent_group/disagreement.csv`, `strict_parent_group/predictions.csv`
  - window/record confusion matrices as both CSV and PNG for each method
  - `overdependence_diagnostics.csv`, `run_manifest.json`, and `README.md`.
- [ ] Implement stable CSV column ordering, UTF-8 BOM for spreadsheet-facing CSVs, readable JSON, Chinese-capable plotting, and a README that separates train-side selection from strict independent verification.
- [ ] Manifest must include timestamps, Git commit/dirty state, command, Python/package versions, source/output hashes, feature/label order, model parameters, fold hash, selection rule, selected method, weight provenance, and an explicit statement that strict data did not select weights or strategy.
- [ ] Write to a new staging sibling; validate all required files and hashes; atomically rename. If target exists, abort without modifying it. Do not delete a failed staging directory automatically; report its path for user-authorized handling.
- [ ] Run focused tests; expect pass.
- [ ] Commit: `git add 程序代码/fusion_weight_study/reporting.py 程序代码/tests/test_fusion_weight_study_reporting.py && git commit -m "feat: report fusion weight experiment"`.

## Task 8: Add the CLI and a fast end-to-end test

**Files:**
- Create: `程序代码/tools/run_channel_fusion_weight_study.py`
- Create: `程序代码/tests/test_channel_fusion_weight_study_cli.py`

- [ ] Write a failing CLI test using tiny synthetic data and a fake probability model so it verifies argument parsing, immutable paths, full artifact creation, and non-overwrite behavior without expensive CatBoost training.
- [ ] Implement explicit CLI flags for three feature files, split file, fold file, source manifest, output directory, model thread count, and optional `--dry-run` validation.
- [ ] Ensure the script inserts the project root into `sys.path`, so it works when launched outside `程序代码`.
- [ ] Run the CLI test from `/private/tmp`; expect pass.
- [ ] From `/private/tmp`, run `python3 '/Users/hewenhao/.codex/worktrees/channel-fusion-weight-study/特征提取/程序代码/tools/run_channel_fusion_weight_study.py' --dry-run --ch3 '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_两种划分_20260920/file_split/features_ch3.csv' --ch4 '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_两种划分_20260920/file_split/features_ch4.csv' --ch5 '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_两种划分_20260920/file_split/features_ch5.csv' --split '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_parent_group严格泛化_20260921/parent_group_split.csv' --folds '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_parent_group严格泛化_20260921/cv_fold_details.csv' --source-manifest '/Users/hewenhao/Documents/特征提取/实验结果/单通道43维_parent_group严格泛化_20260921/run_manifest.json' --output '/Users/hewenhao/.codex/worktrees/channel-fusion-weight-study/特征提取/实验结果/三通道融合权重实验_20260927'`; expect contract validation only and no output directory.
- [ ] Commit: `git add 程序代码/tools/run_channel_fusion_weight_study.py 程序代码/tests/test_channel_fusion_weight_study_cli.py && git commit -m "feat: add fusion study runner"`.

## Task 9: Full verification before the expensive run

- [ ] Run `cd '程序代码' && pytest -q`; all existing and new tests must pass.
- [ ] Run `git diff --check` and confirm no whitespace errors.
- [ ] Record SHA-256 snapshots for the old strict result manifest/models and V3 deployment manifests/models.
- [ ] Run the real CLI dry-run and inspect its audit summary: 43 features, six labels, CH3/4/5 alignment, 68 train-dev parents, 17 strict parents, three folds, no overlap, exact source hashes.
- [ ] Commit any test-only corrections separately; do not weaken contract checks to make malformed inputs pass.

## Task 10: Execute the real experiment

- [ ] Run the real CLI once with the fixed source paths and new output directory. Save stdout/stderr to `run.log` inside the staging/result directory through the reporter, not shell redirection.
- [ ] Do not interrupt a fold mid-write; progress logging must identify channel, fold, fit/validation counts, and elapsed time.
- [ ] If training fails, preserve the staging directory, diagnose, add a regression test, fix, and rerun to a new output path only after user approval if a completed target already exists.
- [ ] Verify all probability coverage, sums, folds, record aggregation, and selection provenance before atomic publication.

## Task 11: Validate results and claims

- [ ] Recompute headline metrics independently from saved prediction CSVs and compare to `fusion_summary.csv` within `1e-12`.
- [ ] Recompute all weights from saved train-side OOF predictions and compare to JSON within `1e-12`.
- [ ] Confirm selected method matches the predeclared train-side rule without consulting strict metrics.
- [ ] Compare train-side improvement and strict improvement to characterize generalization/possible overfit. Phrase strict performance only as this held-out experiment, never as performance of full-data deployment models.
- [ ] Recompute post-run hashes of old strict and V3 deployment assets and prove they match pre-run snapshots.
- [ ] Open generated confusion PNGs and inspect legibility; reopen all CSV/JSON files.
- [ ] Run the full test suite again and `git status --short`.
- [ ] Commit experiment code/report fixes only. Keep generated experiment results out of source commits if ignored; report their absolute path.

## Task 12: Deliver the decision without changing software

- [ ] Present one compact comparison table containing train-side and strict window/record Macro-F1 for equal/global/classwise methods.
- [ ] Present final global and classwise F1 weights, selected strategy, channel-dependence diagnostics, and whether strict improvement is directionally consistent.
- [ ] State clearly whether there is evidence of overfitting and distinguish evidence from uncertainty.
- [ ] Recommend equal weighting if differences are within `0.005`; otherwise recommend the train-selected winner, subject to the strict result being validation only.
- [ ] Explicitly state: no formal software logic changed, no model/feature/preprocessing parameter changed, no old result was overwritten, and no automatic merge was performed.
- [ ] Stop. If software integration is warranted, request a separate explicit instruction and create a separate commit for it.

## Final acceptance checklist

- [ ] Existing `pump_fault_app` code and deployment/evaluation assets are byte-for-byte unchanged.
- [ ] No strict parent/group/sample influenced fold weights, final weights, or strategy selection.
- [ ] No parent appears on both sides of a training/validation or train-dev/strict boundary.
- [ ] All methods are probability-based and support 1/2/3 channels.
- [ ] Classwise weights use class F1, as confirmed by the user.
- [ ] All required artifacts, manifests, matrices, metrics, disagreements, and diagnostics exist and reopen.
- [ ] Tests pass and the experiment is reproducible from one documented command.
- [ ] Branch remains unmerged.
