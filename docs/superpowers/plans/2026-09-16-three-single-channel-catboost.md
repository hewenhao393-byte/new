# Three Single-Channel CatBoost Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate independent channel 3, channel 4, and channel 5 CatBoost six-class models using one deterministic 12-second-record split with no record/window leakage.

**Architecture:** Build one immutable record assignment from the existing 63-feature table, stratified within every source CSV file so each covered fault setting contributes records to both subsets. Reuse the assignment for all channels, select exactly 21 channel-prefixed features per model, and save separate models, metadata, probabilities, metrics, and an aggregate comparison without changing existing artifacts.

**Tech Stack:** Python 3.9, pandas, NumPy, scikit-learn metrics, CatBoost, pytest.

---

## File Structure

- Create `tools/record_level_single_channel_catboost.py`: split construction, leakage validation, training, record fusion, metric persistence, and CLI entrypoint.
- Create `tools/test_record_level_single_channel_catboost.py`: unit tests for deterministic splitting, sibling-channel binding, feature contracts, leakage rejection, and record probability fusion.
- Create `实验结果/三通道单模型_CatBoost_记录级划分/`: generated assignment, three model bundles, prediction tables, metrics, hashes, and explanatory report.
- Preserve `实验结果/三通道融合_文档流程_5Hz/` unchanged as the source and existing pressure-test evidence.

### Task 1: Define and test the record-level split contract

**Files:**
- Create: `tools/test_record_level_single_channel_catboost.py`
- Create: `tools/record_level_single_channel_catboost.py`

- [ ] **Step 1: Write failing tests for a deterministic per-file record split**

```python
def test_split_keeps_every_record_and_is_deterministic(example_records):
    first = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    second = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    assert first.equals(second)
    assert first.group_id.is_unique
    assert set(first.group_id) == set(example_records.group_id)

def test_each_eligible_file_contributes_train_and_test_records(example_records):
    assignment = build_record_assignment(example_records, test_fraction=0.2, seed=42)
    counts = assignment.groupby(['ch4_path', 'subset']).group_id.nunique().unstack(fill_value=0)
    assert (counts['train'] > 0).all()
    assert (counts['test'] > 0).all()
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: FAIL because `record_level_single_channel_catboost` does not yet exist.

- [ ] **Step 3: Implement the minimal deterministic split**

```python
def build_record_assignment(records, test_fraction=0.2, seed=42):
    required = {'group_id', 'record_column', 'label', 'device_id', 'speed_percent',
                'ch3_path', 'ch4_path', 'ch5_path'}
    if missing := required.difference(records.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    unique = records[list(required)].drop_duplicates('group_id').copy()
    if unique.group_id.duplicated().any():
        raise ValueError('group_id metadata is not unique')
    parts = []
    for source_path, group in unique.groupby('ch4_path', sort=True):
        if len(group) < 2:
            raise ValueError(f'File has fewer than two records: {source_path}')
        local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{source_path}'.encode()).digest()[:8], 'big')
        shuffled = group.sample(frac=1, random_state=local_seed % (2**32 - 1)).copy()
        test_count = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        shuffled['subset'] = 'train'
        shuffled.iloc[:test_count, shuffled.columns.get_loc('subset')] = 'test'
        parts.append(shuffled)
    return pd.concat(parts, ignore_index=True).sort_values('group_id').reset_index(drop=True)
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: PASS for deterministic split tests.

- [ ] **Step 5: Commit the split contract**

```bash
git add tools/record_level_single_channel_catboost.py tools/test_record_level_single_channel_catboost.py
git commit -m "Add leakage-safe record split contract"
```

### Task 2: Add hard leakage and channel-alignment validation

**Files:**
- Modify: `tools/test_record_level_single_channel_catboost.py`
- Modify: `tools/record_level_single_channel_catboost.py`

- [ ] **Step 1: Write failing tests for record/window isolation and aligned channel paths**

```python
def test_validate_split_rejects_group_leakage(example_windows, example_assignment):
    broken = pd.concat([example_assignment, example_assignment.iloc[[0]].assign(subset='test')])
    with pytest.raises(ValueError, match='group'):
        validate_no_leakage(example_windows, broken)

def test_validate_split_rejects_misaligned_sibling_channels(example_windows, example_assignment):
    broken = example_windows.copy()
    broken.loc[broken.index[0], 'ch3_path'] = '/wrong/channel3.csv'
    with pytest.raises(ValueError, match='channel metadata'):
        validate_no_leakage(broken, example_assignment)
```

- [ ] **Step 2: Run the new tests and verify both fail because validation is missing**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: FAIL with `validate_no_leakage` missing.

- [ ] **Step 3: Implement validation with explicit assertions**

```python
def validate_no_leakage(windows, assignment):
    if assignment.group_id.duplicated().any():
        raise ValueError('Duplicate group assignment causes group leakage')
    merged = windows.merge(assignment[['group_id', 'subset']], on='group_id', validate='many_to_one')
    if len(merged) != len(windows):
        raise ValueError('Some windows have no record assignment')
    if windows.duplicated(['group_id', 'window_id']).any():
        raise ValueError('Duplicate window key')
    per_group = windows.groupby('group_id')[['ch3_path', 'ch4_path', 'ch5_path']].nunique()
    if not per_group.eq(1).all().all():
        raise ValueError('Inconsistent channel metadata within group')
    train_ids = set(assignment.loc[assignment.subset.eq('train'), 'group_id'])
    test_ids = set(assignment.loc[assignment.subset.eq('test'), 'group_id'])
    if train_ids & test_ids:
        raise ValueError('group leakage detected')
    return merged
```

- [ ] **Step 4: Add an output audit with counts by class, device, speed, file, and subset**

The script must write `split_distribution.csv` using:

```python
assignment.groupby(['subset', 'label', 'device_id', 'speed_percent'], dropna=False).agg(
    records=('group_id', 'nunique'), files=('ch4_path', 'nunique')
).reset_index()
```

- [ ] **Step 5: Run focused tests and commit**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: all split and validation tests PASS.

```bash
git add tools/record_level_single_channel_catboost.py tools/test_record_level_single_channel_catboost.py
git commit -m "Validate record and channel isolation"
```

### Task 3: Add the fixed single-channel training and evaluation pipeline

**Files:**
- Modify: `tools/test_record_level_single_channel_catboost.py`
- Modify: `tools/record_level_single_channel_catboost.py`

- [ ] **Step 1: Write failing tests for exact 21-feature contracts and probability fusion**

```python
@pytest.mark.parametrize('channel', ['3', '4', '5'])
def test_channel_features_returns_exact_order(channel, formal_feature_names):
    selected = select_channel_features(formal_feature_names, channel)
    assert len(selected) == 21
    assert selected == [n for n in formal_feature_names if n.startswith(f'ch{channel}_')]

def test_fuse_record_probabilities_averages_windows():
    frame = pd.DataFrame({'group_id': ['a', 'a'], '正常': [0.8, 0.6], '汽蚀': [0.2, 0.4]})
    fused = fuse_record_probabilities(frame, ['正常', '汽蚀'])
    assert fused.loc['a', '正常'] == pytest.approx(0.7)
    assert fused.loc['a', 'predicted_label'] == '正常'
```

- [ ] **Step 2: Run tests and verify failures for missing functions**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: FAIL because selection and fusion functions are missing.

- [ ] **Step 3: Implement exact feature selection and record fusion**

```python
def select_channel_features(names, channel):
    selected = [name for name in names if name.startswith(f'ch{channel}_')]
    if len(selected) != 21 or len(set(selected)) != 21:
        raise ValueError(f'Channel {channel} does not have 21 unique features')
    return selected

def fuse_record_probabilities(probability_frame, classes):
    fused = probability_frame.groupby('group_id')[classes].mean()
    fused['predicted_label'] = fused[classes].idxmax(axis=1)
    return fused
```

- [ ] **Step 4: Implement one fixed training function shared by all channels**

```python
FIXED_PARAMS = {
    'loss_function': 'MultiClass', 'iterations': 540, 'depth': 8,
    'learning_rate': 0.05, 'l2_leaf_reg': 100, 'random_strength': 5,
    'rsm': 0.7, 'auto_class_weights': 'SqrtBalanced',
    'random_seed': 42, 'thread_count': 4, 'allow_writing_files': False,
}

def fit_channel_model(train, test, features, channel):
    model = CatBoostClassifier(**FIXED_PARAMS)
    model.fit(Pool(train[features], train.label), verbose=180)
    probabilities = model.predict_proba(Pool(test[features], test.label), thread_count=4)
    return model, probabilities
```

The orchestration loop must train channels `3`, `4`, and `5` against the same assignment, save `chN_catboost.cbm`, reload it, and assert prediction probabilities are equal within `atol=1e-12`.

- [ ] **Step 5: Persist complete metrics and predictions**

For every channel write:

- `chN_metadata.json`: 21 features, classes, parameters, input hashes, split hash.
- `chN_window_predictions.csv`: group/window IDs, true/predicted labels, six probabilities.
- `chN_record_predictions.csv`: group ID, true/predicted labels, averaged six probabilities.
- `chN_window_metrics.json` and `chN_record_metrics.json`: accuracy, balanced accuracy, Macro-F1, classification report, confusion matrix.

- [ ] **Step 6: Run unit tests and commit**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: all tests PASS.

```bash
git add tools/record_level_single_channel_catboost.py tools/test_record_level_single_channel_catboost.py
git commit -m "Train fixed single-channel CatBoost models"
```

### Task 4: Execute the experiment without overwriting prior outputs

**Files:**
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/record_assignment.csv`
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/split_distribution.csv`
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/ch3_*`
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/ch4_*`
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/ch5_*`
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/comparison.csv`

- [ ] **Step 1: Run the experiment once**

Run: `.venv/bin/python tools/record_level_single_channel_catboost.py`

Expected: process exits 0 after printing three channel result summaries. If the output directory already exists, it must stop without overwriting.

- [ ] **Step 2: Independently verify split integrity**

Run:

```bash
.venv/bin/python -c "import pandas as pd; a=pd.read_csv('实验结果/三通道单模型_CatBoost_记录级划分/record_assignment.csv'); assert a.group_id.is_unique; assert set(a.subset)=={'train','test'}; assert not set(a[a.subset=='train'].group_id)&set(a[a.subset=='test'].group_id); print(a.groupby('subset').group_id.nunique())"
```

Expected: nonzero train/test counts and no assertion error.

- [ ] **Step 3: Verify all three prediction tables cover the identical test records**

Run:

```bash
.venv/bin/python -c "import pandas as pd; p='实验结果/三通道单模型_CatBoost_记录级划分'; ids=[set(pd.read_csv(f'{p}/ch{c}_record_predictions.csv').group_id) for c in '345']; assert ids[0]==ids[1]==ids[2]; print(len(ids[0]))"
```

Expected: one positive count and no assertion error.

- [ ] **Step 4: Verify source and assignment hashes recorded in all metadata files**

Run: `.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py -v`

Expected: all tests PASS and no unexpected warning from the experiment module.

- [ ] **Step 5: Commit the generated audit artifacts and code**

```bash
git add tools/record_level_single_channel_catboost.py tools/test_record_level_single_channel_catboost.py 实验结果/三通道单模型_CatBoost_记录级划分
git commit -m "Evaluate three record-split channel models"
```

### Task 5: Produce the scientific interpretation report

**Files:**
- Create: `实验结果/三通道单模型_CatBoost_记录级划分/实验说明.md`

- [ ] **Step 1: Write a report grounded only in generated metrics**

The report must include:

- Exact record and window counts by subset.
- Confirmation that the same group IDs were used for all channels.
- Channel-to-sensor-location mapping.
- Accuracy, Balanced Accuracy, Macro-F1, and six class recalls for each channel.
- Confusion analysis for normal, looseness, and bearing fault.
- A side-by-side note that the existing file-holdout experiment is a harder pressure test with a different evaluation target.
- The limitation that records from the same source file may be correlated because acquisition independence is undocumented.
- No claim of cross-device generalization and no promise of 90% accuracy.

- [ ] **Step 2: Run a placeholder and provenance scan**

Run: `rg -n 'TBD|TODO|待补|待定|保证达到|完全泛化' 实验结果/三通道单模型_CatBoost_记录级划分/实验说明.md`

Expected: no matches.

- [ ] **Step 3: Run the final verification suite**

Run:

```bash
.venv/bin/python -m pytest tools/test_record_level_single_channel_catboost.py tools/test_catboost345.py tools/test_channel_ablation.py tools/test_tune345_regularized.py tools/test_feature_group_ablation.py -q
```

Expected: all tests PASS, exit code 0.

- [ ] **Step 4: Confirm no existing source artifact changed**

Compare the source hashes in the three metadata files with the current SHA256 of `train_features_63.csv` and `window_metadata.csv`; expected: exact match. Check `git status --short` and verify no existing model or feature CSV was modified by this plan.

- [ ] **Step 5: Commit the report**

```bash
git add 实验结果/三通道单模型_CatBoost_记录级划分/实验说明.md
git commit -m "Document single-channel CatBoost results"
```
