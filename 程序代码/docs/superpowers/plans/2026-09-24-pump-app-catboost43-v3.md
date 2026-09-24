# Pump Fault App CatBoost43 V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deployed 21-feature BP inference path with one formal V3 contract that runs independent 43-feature CatBoost inference for CH3/CH4/CH5 and equal-weight probability fusion across valid channels.

**Architecture:** Preserve the existing `pump_fault_app` package and UI shell, but replace its formal domain, feature, prediction, fusion, and inference internals. All consumers receive one immutable `MultiChannelDiagnosisResult`; single-run, batch, history, reporting, CLI, and UI must not independently recompute labels or probabilities. Evaluation artifacts remain untouched, while three full-data deployment models and a deployment-only manifest live under `程序代码/models/catboost43_v3/`.

**Tech Stack:** Python 3.9+, NumPy, pandas, SciPy, PyWavelets, CatBoost, Streamlit, python-docx, SQLite, pytest.

---

## File map

**Create:**

- `pump_fault_app/domain/diagnosis_models.py`: immutable V3 request/result types and invariants.
- `pump_fault_app/feature_extraction/catboost43_features.py`: accepted 43-feature implementation.
- `pump_fault_app/prediction/catboost_loader.py`: deployment manifest and model validation.
- `pump_fault_app/prediction/channel_predictor.py`: one-window CatBoost probability prediction.
- `pump_fault_app/fusion/window_fusion.py`: window-to-channel probability mean.
- `pump_fault_app/fusion/channel_fusion.py`: valid-channel equal-weight fusion and agreement.
- `pump_fault_app/inference/channel_inference.py`: one channel's full pipeline.
- `pump_fault_app/inference/multichannel_inference.py`: strict three-file coordination.
- `pump_fault_app/history/v3_models.py`, `v3_repository.py`: V3 persistence boundary.
- `tools/train_catboost43_v3_deployment.py`: reproducible all-data training and publication.
- `tests/test_pump_fault_app_v3_models.py`, `test_pump_fault_app_channel_fusion.py`, `test_pump_fault_app_multichannel_input.py`, `test_catboost43_v3_deployment_training.py`: focused V3 coverage.
- `tests/test_pump_fault_app_no_legacy.py`: deployed-source dead-reference guard.
- `docs/pump_fault_app_v3_model_provenance.md`: model-use boundary.

**Replace or modify:**

- `pump_fault_app/domain/formal_contract.py`, `version.py`, `domain/labels.py`, `domain/records.py`.
- `preprocessing/signal_preprocessing.py`, `windowing/segmenter.py`, `quality/signal_quality.py`.
- `inference/visualization.py`, `services/*.py`, `batch/*.py`, `history/*.py`, `reporting/*.py`, `export/*.py`.
- `presentation/*.py`, `ui/streamlit_app.py`, `ui/pages/*.py`, `app/*.py`, package `__init__.py` files.
- Current application tests and operational documentation.

**Delete only after a fresh user-confirmed deletion review:**

- `feature_extraction/extractor.py`, `prediction/predictor.py`, `fusion/probability_fusion.py`, `inference/service.py`.
- `model_training/`, `model_registry/`, `sample_repository/`.
- `ui/pages/model_optimization.py`, `tests/test_pump_fault_app_model_optimization.py`.

## Task 1: Freeze the V3 contract and label mapping

**Files:**

- Modify: `pump_fault_app/version.py`
- Replace: `pump_fault_app/domain/formal_contract.py`
- Modify: `pump_fault_app/domain/labels.py`
- Rewrite test: `tests/test_inference_contract.py`

- [ ] **Step 1: Write failing V3 contract tests**

```python
from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER, FORMAL_V3_CONTRACT
from pump_fault_app.domain.labels import display_label
from pump_fault_app.version import APP_VERSION, FEATURE_VERSION, INFERENCE_CONTRACT_VERSION, MODEL_VERSION


def test_v3_contract_is_the_only_deployed_contract():
    assert APP_VERSION == "pump-fault-app-v3"
    assert MODEL_VERSION == "catboost43-six-class-v3"
    assert FEATURE_VERSION == "43-feature-v1"
    assert INFERENCE_CONTRACT_VERSION == "formal-V3-catboost43"
    assert FORMAL_V3_CONTRACT.signal.target_sampling_rate == 12_000
    assert FORMAL_V3_CONTRACT.signal.filter_low_hz == 5.0
    assert FORMAL_V3_CONTRACT.signal.filter_high_hz == 5_000.0
    assert FORMAL_V3_CONTRACT.signal.window_size == 4_800
    assert FORMAL_V3_CONTRACT.signal.step_size == 2_400
    assert FORMAL_V3_CONTRACT.channels == ("CH3", "CH4", "CH5")
    assert len(FORMAL_FEATURE_NAMES) == 43
    assert FORMAL_LABEL_ORDER == ("正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀")
    assert display_label("松动") == "机械松动"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_inference_contract.py -q`

Expected: FAIL because `FORMAL_V3_CONTRACT` and the V3 version strings do not exist.

- [ ] **Step 3: Implement the frozen contract**

```python
# version.py
APP_VERSION = "pump-fault-app-v3"
MODEL_VERSION = "catboost43-six-class-v3"
FEATURE_VERSION = "43-feature-v1"
INFERENCE_CONTRACT_VERSION = "formal-V3-catboost43"

# labels.py
DISPLAY_LABELS = {"松动": "机械松动"}

def display_label(label: str) -> str:
    assert_valid_label(label)
    return DISPLAY_LABELS.get(label, label)
```

Define `FORMAL_FEATURE_NAMES` exactly as the 43-name tuple in the approved design, define `SignalContract(12000, 5.0, 5000.0, 4800, 2400, "db6", 3)`, and expose only `FORMAL_V3_CONTRACT`. Its `validate()` must reject a non-43 feature tuple, reordered names, unsupported channels, or any label order other than `FORMAL_LABEL_ORDER`.

- [ ] **Step 4: Verify GREEN and scan for accidental V2 exports**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_inference_contract.py -q`

Run: `rg -n "FORMAL_V2_CONTRACT|formal-V2" pump_fault_app --glob '*.py'`

Expected: contract tests PASS; the scan lists migration sites to be handled by later tasks, but `formal_contract.py` itself exports no V2 object.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/version.py pump_fault_app/domain/formal_contract.py pump_fault_app/domain/labels.py tests/test_inference_contract.py
git commit -m "feat: freeze CatBoost43 V3 inference contract"
```

## Task 2: Add immutable V3 request and result models

**Files:**

- Create: `pump_fault_app/domain/diagnosis_models.py`
- Modify: `pump_fault_app/domain/records.py`
- Create test: `tests/test_pump_fault_app_v3_models.py`

- [ ] **Step 1: Write failing invariant tests**

```python
import pytest
from pathlib import Path
from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest, probability_tuple


def test_request_rejects_duplicate_channels():
    item = ChannelInput("CH3", Path("a.csv"), "signal", None)
    with pytest.raises(ValueError, match="duplicate channel"):
        MultiChannelInferenceRequest((item, item), 12_000, 1500.0)


def test_probability_tuple_requires_six_normalized_values():
    assert probability_tuple([0.1, 0.2, 0.3, 0.1, 0.2, 0.1]) == (0.1, 0.2, 0.3, 0.1, 0.2, 0.1)
    with pytest.raises(ValueError, match="sum to 1"):
        probability_tuple([0.1] * 6)
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_v3_models.py -q`

Expected: collection FAIL because `diagnosis_models` does not exist.

- [ ] **Step 3: Implement the public models and invariants**

```python
ChannelName = Literal["CH3", "CH4", "CH5"]

@dataclass(frozen=True)
class ChannelInput:
    channel: ChannelName
    file_path: Path
    signal_column: str | None = None
    time_column: str | None = None

@dataclass(frozen=True)
class MultiChannelInferenceRequest:
    channels: tuple[ChannelInput, ...]
    sampling_rate_hz: int
    rpm: float
    model_directory: Path | None = None

    def __post_init__(self):
        names = tuple(item.channel for item in self.channels)
        if not 1 <= len(names) <= 3:
            raise ValueError("channels must contain 1 to 3 inputs")
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel input")
        if self.sampling_rate_hz <= 0 or self.rpm <= 0:
            raise ValueError("sampling rate and rpm must be positive")
```

Add complete `WindowDiagnosisResult`, `ChannelDiagnosisResult`, and `MultiChannelDiagnosisResult` dataclasses from the approved design. Enforce valid/invalid and diagnosed/failed field combinations in `__post_init__`. Store probabilities as tuples in formal label order.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_v3_models.py tests/test_pump_fault_app_records.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/domain/diagnosis_models.py pump_fault_app/domain/records.py tests/test_pump_fault_app_v3_models.py
git commit -m "feat: define V3 multichannel diagnosis results"
```

## Task 3: Port and lock the accepted 43-feature extractor

**Files:**

- Create: `pump_fault_app/feature_extraction/catboost43_features.py`
- Modify: `pump_fault_app/feature_extraction/__init__.py`
- Rewrite test: `tests/test_pump_fault_app_feature_extraction.py`

- [ ] **Step 1: Write failing count, order, and reference-parity tests**

```python
import numpy as np
from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES
from pump_fault_app.feature_extraction import extract_catboost43_features


def test_catboost43_feature_count_and_order():
    t = np.arange(4_800) / 12_000.0
    signal = np.sin(2 * np.pi * 25 * t) + 0.25 * np.sin(2 * np.pi * 50 * t)
    vector = extract_catboost43_features(signal, rpm=1500.0)
    assert vector.feature_names == FORMAL_FEATURE_NAMES
    assert len(vector.values) == 43
    assert np.isfinite(vector.values).all()
```

Add a parity test that loads one accepted `features_ch3.csv` row, reconstructs its source 4800-sample window through the accepted manifest metadata, and compares all 43 values using the tolerances recorded by feature acceptance. The test must skip with an explicit reason only when the external source dataset is absent; it must not silently pass on a mismatch.

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_feature_extraction.py -q`

Expected: FAIL because `extract_catboost43_features` does not exist.

- [ ] **Step 3: Port the accepted implementation without redefining formulas**

Use `/Users/hewenhao/.codex/worktrees/single-channel-43-features/特征提取/feature_pipeline/features.py` as the source of truth. Preserve its order-band width, FFT window, band boundaries, db6 level-3 packet order, 1000–5000 Hz envelope preprocessing, peak rules, epsilon, finite-value validation, and final tuple-order assertion. Replace its global config dependency with `FORMAL_V3_CONTRACT` values.

```python
def extract_catboost43_features(samples: np.ndarray, rpm: float) -> FeatureVector:
    values_by_name = _extract_accepted_feature_map(np.asarray(samples, dtype=np.float64), rpm)
    if tuple(values_by_name) != FORMAL_FEATURE_NAMES:
        raise RuntimeError("feature order mismatch")
    values = tuple(float(values_by_name[name]) for name in FORMAL_FEATURE_NAMES)
    if not np.isfinite(values).all():
        raise ValueError("non-finite feature")
    return FeatureVector(FORMAL_FEATURE_NAMES, values, SampleMetadata(rpm=rpm))
```

- [ ] **Step 4: Verify GREEN and parity**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_feature_extraction.py -q`

Expected: PASS with exactly 43 ordered finite features and accepted-row parity.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/feature_extraction/catboost43_features.py pump_fault_app/feature_extraction/__init__.py tests/test_pump_fault_app_feature_extraction.py
git commit -m "feat: port accepted 43-feature extractor"
```

## Task 4: Migrate quality, preprocessing, windowing, and visualization to V3

**Files:**

- Modify: `pump_fault_app/quality/signal_quality.py`
- Modify: `pump_fault_app/preprocessing/signal_preprocessing.py`
- Modify: `pump_fault_app/windowing/segmenter.py`
- Modify: `pump_fault_app/inference/visualization.py`
- Rewrite tests: `tests/test_pump_fault_app_signal_quality.py`, `test_pump_fault_app_preprocessing.py`, `test_pump_fault_app_windowing.py`

- [ ] **Step 1: Write failing V3 signal-processing tests**

```python
def test_windowing_uses_v3_4800_2400_contract(preprocessed_record):
    result = segment_preprocessed_signal(preprocessed_record)
    assert result.window_size == 4_800
    assert result.step_size == 2_400
    assert [w.start_index for w in result.windows[:3]] == [0, 2_400, 4_800]


def test_preprocessing_uses_5_to_5000_hz_zero_phase_filter(raw_record):
    result = preprocess_raw_signal(raw_record)
    assert result.target_sampling_rate_hz == 12_000
    assert "5-5000 Hz zero-phase bandpass" in result.processing_log
```

Add rejection tests for all-zero, NaN/Inf, severe clipping, and fewer than 4800 processed samples.

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_signal_quality.py tests/test_pump_fault_app_preprocessing.py tests/test_pump_fault_app_windowing.py -q`

Expected: FAIL on old 10 Hz, 2400, and 1200 values.

- [ ] **Step 3: Replace V2 imports with `FORMAL_V3_CONTRACT`**

Keep the current anti-alias `resample_poly` path and zero-phase SOS filtering, but read every cutoff/window value from the V3 contract. Visualization must accept a channel result and generate four plots from that channel only.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_signal_quality.py tests/test_pump_fault_app_preprocessing.py tests/test_pump_fault_app_windowing.py tests/test_pump_fault_app_inference.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/quality pump_fault_app/preprocessing pump_fault_app/windowing pump_fault_app/inference/visualization.py tests/test_pump_fault_app_signal_quality.py tests/test_pump_fault_app_preprocessing.py tests/test_pump_fault_app_windowing.py
git commit -m "feat: migrate signal processing to V3 contract"
```

## Task 5: Implement probability-only window and channel fusion

**Files:**

- Create: `pump_fault_app/fusion/window_fusion.py`
- Create: `pump_fault_app/fusion/channel_fusion.py`
- Modify: `pump_fault_app/fusion/__init__.py`
- Create test: `tests/test_pump_fault_app_channel_fusion.py`
- Rewrite test: `tests/test_pump_fault_app_fusion.py`

- [ ] **Step 1: Write failing fusion tests**

```python
def test_two_channel_fusion_is_equal_probability_mean(ch3_result, ch4_result):
    result = fuse_valid_channels((ch3_result, ch4_result))
    expected = tuple((a + b) / 2 for a, b in zip(ch3_result.class_probabilities, ch4_result.class_probabilities))
    assert result.fused_probabilities == pytest.approx(expected)


def test_invalid_channel_is_excluded(ch3_invalid, ch4_result, ch5_result):
    result = fuse_valid_channels((ch3_invalid, ch4_result, ch5_result))
    assert result.valid_channels == ("CH4", "CH5")


def test_all_invalid_channels_return_failed(ch3_invalid, ch4_invalid, ch5_invalid):
    result = fuse_valid_channels((ch3_invalid, ch4_invalid, ch5_invalid))
    assert result.status == "failed"
    assert result.predicted_label is None
    assert result.fused_probabilities is None
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_fusion.py tests/test_pump_fault_app_channel_fusion.py -q`

Expected: FAIL because V3 fusion modules do not exist.

- [ ] **Step 3: Implement arithmetic mean and agreement rules**

```python
def mean_probabilities(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FORMAL_LABEL_ORDER):
        raise ValueError("probability matrix must have six columns")
    return probability_tuple(matrix.mean(axis=0))


def agreement(labels: tuple[str, ...]) -> tuple[str | None, int]:
    count = Counter(labels).most_common(1)[0][1]
    if len(labels) == 1:
        return None, 1
    if count == len(labels):
        return "高一致", count
    if len(labels) == 3 and count == 2:
        return "中等一致", count
    return "低一致", count
```

Do not expose hard-vote fusion; labels are used only to describe agreement after probability fusion.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_fusion.py tests/test_pump_fault_app_channel_fusion.py -q`

Expected: PASS including probability-sum and 1/2/3-channel cases.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/fusion tests/test_pump_fault_app_fusion.py tests/test_pump_fault_app_channel_fusion.py
git commit -m "feat: add window and channel probability fusion"
```

## Task 6: Build and validate deployment model assets

**Files:**

- Create: `tools/train_catboost43_v3_deployment.py`
- Create: `tests/test_catboost43_v3_deployment_training.py`
- Create after training: `models/catboost43_v3/ch3.cbm`, `ch4.cbm`, `ch5.cbm`, `manifest.json`
- Create: `docs/pump_fault_app_v3_model_provenance.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing source and publication tests**

```python
def test_training_uses_only_file_split_tables():
    assert DEPLOYMENT_SOURCES == {
        "CH3": SOURCE_ROOT / "file_split/features_ch3.csv",
        "CH4": SOURCE_ROOT / "file_split/features_ch4.csv",
        "CH5": SOURCE_ROOT / "file_split/features_ch5.csv",
    }
    assert all("temporal_split" not in str(path) for path in DEPLOYMENT_SOURCES.values())


def test_fixed_params_are_p1_at_500():
    assert FIXED_PARAMS["iterations"] == 500
    assert FIXED_PARAMS["depth"] == 8
    assert FIXED_PARAMS["learning_rate"] == 0.05
    assert FIXED_PARAMS["l2_leaf_reg"] == 100
    assert FIXED_PARAMS["random_strength"] == 5
    assert FIXED_PARAMS["rsm"] == 0.7
    assert FIXED_PARAMS["auto_class_weights"] == "SqrtBalanced"
    assert FIXED_PARAMS["random_seed"] == 2026
```

Add a fake-CatBoost test proving three models are fit with all rows and that manifest publication is atomic only after reload, feature-name, class, probability, and SHA-256 checks pass.

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_catboost43_v3_deployment_training.py -q`

Expected: FAIL because the deployment trainer does not exist.

- [ ] **Step 3: Implement validated full-data training**

The script must load each CSV separately, require exact metadata plus `FORMAL_FEATURE_NAMES`, reject duplicate `(record_id, window_id)` keys, reject NaN/Inf, require the six-label set, and assert sibling-channel key alignment. Fit `CatBoostClassifier(**FIXED_PARAMS)` per channel with columns in exact contract order. Write models to a temporary sibling directory, reload them, validate features/classes/probabilities, write manifest, fsync where available, then atomically rename the completed directory to `models/catboost43_v3`.

Add `catboost>=1.2,<2.0` to `requirements.txt`; the deployed path must not add a scaler or imputer.

```python
def train_channel(frame: pd.DataFrame, channel: str) -> CatBoostClassifier:
    x = frame.loc[:, FORMAL_FEATURE_NAMES]
    y = frame["label"]
    model = CatBoostClassifier(**FIXED_PARAMS)
    model.fit(Pool(x, label=y, feature_names=list(FORMAL_FEATURE_NAMES)))
    if tuple(model.feature_names_) != FORMAL_FEATURE_NAMES:
        raise RuntimeError(f"{channel} feature order mismatch")
    return model
```

- [ ] **Step 4: Verify unit tests before the long run**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_catboost43_v3_deployment_training.py -q`

Expected: PASS without training the real models.

- [ ] **Step 5: Train the three real deployment models**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python tools/train_catboost43_v3_deployment.py --source-root '../实验结果/单通道43维_两种划分_20260920' --output-dir models/catboost43_v3`

Expected: three independently reloaded models, one manifest, exact 43-feature order, six classes, normalized probabilities, and no writes under `单通道43维_parent_group严格泛化_20260921/`.

- [ ] **Step 6: Record provenance and commit**

The provenance document must state: full accepted-data retraining, software-inference-only use, no held-out deployment metric, no reuse of parent_group claims, and future evaluation on new independent data.

```bash
git add requirements.txt tools/train_catboost43_v3_deployment.py tests/test_catboost43_v3_deployment_training.py models/catboost43_v3 docs/pump_fault_app_v3_model_provenance.md
git commit -m "feat: publish full-data CatBoost43 deployment models"
```

## Task 7: Load channel models and predict one window

**Files:**

- Create: `pump_fault_app/prediction/catboost_loader.py`
- Create: `pump_fault_app/prediction/channel_predictor.py`
- Modify: `pump_fault_app/prediction/__init__.py`
- Rewrite test: `tests/test_pump_fault_app_prediction.py`

- [ ] **Step 1: Write failing loader and class-reordering tests**

```python
def test_loader_rejects_manifest_feature_reordering(tmp_path):
    model_dir = write_fake_model_dir(tmp_path, features=tuple(reversed(FORMAL_FEATURE_NAMES)))
    with pytest.raises(ValueError, match="feature order"):
        load_catboost43_models(model_dir)


def test_predictor_reorders_model_classes_to_formal_order(fake_model):
    fake_model.classes_ = np.array(["松动", "正常", "汽蚀", "联轴器不对中", "转子不平衡", "轴承故障"])
    result = predict_channel_window(feature_vector(), fake_model, "CH3")
    assert tuple(result.class_probabilities) == pytest.approx(probabilities_in_formal_order)
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_prediction.py -q`

Expected: FAIL because CatBoost V3 loader and predictor do not exist.

- [ ] **Step 3: Implement manifest/hash/model validation**

`load_catboost43_models()` must require three declared model files, verify file hashes, `training_scope`, `intended_use`, versions, feature order, and label set, then load each with `CatBoostClassifier().load_model()`. `predict_channel_window()` must build a DataFrame in exact feature order, call `predict_proba`, reorder by `classes_`, validate six finite normalized values, and return `WindowDiagnosisResult`.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_prediction.py -q`

Expected: PASS for CH3, CH4, CH5 and every rejection case.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/prediction tests/test_pump_fault_app_prediction.py
git commit -m "feat: load and validate channel CatBoost models"
```

## Task 8: Implement strict multichannel input and inference

**Files:**

- Create: `pump_fault_app/inference/channel_inference.py`
- Create: `pump_fault_app/inference/multichannel_inference.py`
- Modify: `pump_fault_app/inference/__init__.py`
- Create test: `tests/test_pump_fault_app_multichannel_input.py`
- Rewrite test: `tests/test_pump_fault_app_inference.py`

- [ ] **Step 1: Write failing strict-input tests**

```python
def test_multichannel_rejects_mixed_time_column_presence(tmp_path):
    request = request_for_files(tmp_path, CH3_time="time", CH4_time=None)
    with pytest.raises(ValueError, match="all provide time columns or none"):
        validate_multichannel_inputs(request)


def test_multichannel_rejects_length_mismatch(tmp_path):
    request = request_with_lengths(tmp_path, CH3=9600, CH4=9599)
    with pytest.raises(ValueError, match="signal lengths differ"):
        validate_multichannel_inputs(request)


def test_one_invalid_channel_is_excluded_but_result_is_diagnosed(valid_ch4, valid_ch5, invalid_ch3):
    result = fuse_valid_channels((invalid_ch3, valid_ch4, valid_ch5))
    assert result.status == "diagnosed"
    assert result.valid_channels == ("CH4", "CH5")
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_multichannel_input.py tests/test_pump_fault_app_inference.py -q`

Expected: FAIL because V3 inference modules do not exist.

- [ ] **Step 3: Implement channel and multichannel orchestration**

`run_channel_inference()` must catch input, quality, preprocessing, windowing, feature, and prediction failures and convert them into an invalid `ChannelDiagnosisResult` without hiding the failure stage. `run_multichannel_inference()` must validate cross-file alignment before per-channel execution, load models once, execute each supplied channel independently, and call only `fuse_valid_channels()` for the final result.

```python
def run_multichannel_inference(request: MultiChannelInferenceRequest) -> MultiChannelDiagnosisResult:
    loaded_inputs = read_and_validate_channel_files(request)
    models = load_catboost43_models(request.model_directory)
    channel_results = tuple(
        run_channel_inference(item, loaded_inputs[item.channel], request, models[item.channel])
        for item in request.channels
    )
    return fuse_valid_channels(channel_results)
```

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_multichannel_input.py tests/test_pump_fault_app_inference.py -q`

Expected: PASS for 1/2/3 inputs, strict alignment, invalid-channel exclusion, and all-invalid failure.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/inference tests/test_pump_fault_app_multichannel_input.py tests/test_pump_fault_app_inference.py
git commit -m "feat: orchestrate strict multichannel V3 inference"
```

## Task 9: Move services, batch, CLI, and self-check onto the single V3 path

**Files:**

- Modify: `pump_fault_app/services/app_service.py`, `report_export_service.py`
- Modify: `pump_fault_app/batch/service.py`, `batch/manifest.py`
- Modify: `pump_fault_app/app/cli.py`, `batch_cli.py`, `bootstrap.py`, `self_check.py`
- Rewrite tests: `test_pump_fault_app_services.py`, `test_pump_fault_app_batch.py`, `test_pump_fault_app_batch_manifest.py`, `test_pump_fault_app_cli.py`, `test_pump_fault_app_batch_cli.py`, `test_pump_fault_app_self_check.py`

- [ ] **Step 1: Write failing shared-path tests**

```python
def test_single_and_batch_services_call_the_same_v3_entrypoint(monkeypatch, request):
    calls = []
    monkeypatch.setattr("pump_fault_app.services.app_service.run_multichannel_inference", lambda item: calls.append(item) or diagnosed_result())
    run_single_diagnosis(app_single_request(request))
    run_batch_diagnosis(app_batch_request(request))
    assert len(calls) == 2
    assert all(isinstance(item, MultiChannelInferenceRequest) for item in calls)
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_services.py tests/test_pump_fault_app_batch.py tests/test_pump_fault_app_cli.py tests/test_pump_fault_app_self_check.py -q`

Expected: FAIL because current services still construct `FormalInferenceRequest` and accept a BP bundle.

- [ ] **Step 3: Replace public requests and manifests**

CLI syntax must accept repeated channel arguments with per-channel file, signal column, and optional time column, plus one sampling rate and rpm. Batch manifest rows must encode 1～3 channel input objects and may not contain `model_bundle_path`. Self-check must validate contract, manifest, three model hashes, model features/classes, and one configured demo request.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_services.py tests/test_pump_fault_app_batch.py tests/test_pump_fault_app_batch_manifest.py tests/test_pump_fault_app_cli.py tests/test_pump_fault_app_batch_cli.py tests/test_pump_fault_app_self_check.py -q`

Expected: PASS with no BP bundle argument or V2 service call.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/services pump_fault_app/batch pump_fault_app/app tests/test_pump_fault_app_services.py tests/test_pump_fault_app_batch.py tests/test_pump_fault_app_batch_manifest.py tests/test_pump_fault_app_cli.py tests/test_pump_fault_app_batch_cli.py tests/test_pump_fault_app_self_check.py
git commit -m "feat: route all application entrypoints through V3"
```

## Task 10: Add V3 history without migrating old BP databases

**Files:**

- Create: `pump_fault_app/history/v3_models.py`, `v3_repository.py`
- Modify: `pump_fault_app/history/service.py`, `history/__init__.py`
- Rewrite test: `tests/test_pump_fault_app_history.py`

- [ ] **Step 1: Write failing V3 persistence tests**

```python
def test_v3_history_saves_channels_probabilities_and_versions(tmp_path, diagnosed_result):
    repo = V3HistoryRepository(tmp_path / "history_v3.sqlite3")
    record = repo.save(diagnosed_result, source_files={"CH3": "a.csv", "CH4": "b.csv"}, sampling_rate_hz=20_000, rpm=1500.0)
    loaded = repo.get(record.id)
    assert loaded.input_channels == ("CH3", "CH4")
    assert loaded.model_version == "catboost43-six-class-v3"
    assert loaded.feature_version == "43-feature-v1"
    assert loaded.contract_version == "formal-V3-catboost43"
    assert loaded.channel_results[0]["class_probabilities"]
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_history.py -q`

Expected: FAIL because V3 history types and schema do not exist.

- [ ] **Step 3: Implement a new V3 database/table**

Use `runtime_data/pump_fault_history_v3.sqlite3` by default. Store indexed scalar fields plus deterministic JSON for source files, channel results, fused probabilities, warnings, and agreement. Do not open or migrate the V2 database. Only diagnosed results produce a report path; failed results may be saved without a report when explicitly requested by the service.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_history.py -q`

Expected: PASS and the V2 fixture remains byte-identical.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/history tests/test_pump_fault_app_history.py
git commit -m "feat: persist V3 multichannel diagnosis history"
```

## Task 11: Rebuild reporting, export, and Word output around V3 results

**Files:**

- Modify: `pump_fault_app/reporting/summary.py`, `view_data.py`, `word_export.py`, `reporting/__init__.py`
- Modify: `pump_fault_app/export/writer.py`, `export/__init__.py`
- Rewrite tests: `tests/test_pump_fault_app_reporting.py`, `test_pump_fault_app_export.py`

- [ ] **Step 1: Write failing report-content tests**

```python
def test_word_report_contains_v3_sections_and_no_bp_terms(tmp_path, diagnosed_result):
    path = write_word_report(diagnosed_result, tmp_path / "report.docx")
    text = "\n".join(p.text for p in Document(path).paragraphs)
    for marker in ("输入通道", "各通道诊断结果", "多通道融合结果", "模型输出概率", "formal-V3-catboost43"):
        assert marker in text
    for forbidden in ("21维", "BP神经网络", "formal-V2", "诊断置信度"):
        assert forbidden not in text
```

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_reporting.py tests/test_pump_fault_app_export.py -q`

Expected: FAIL on old BP/21-feature fields.

- [ ] **Step 3: Implement V3-only summary and report views**

Build all tables from `MultiChannelDiagnosisResult`. Display “机械松动” through `display_label()` only. Show final fused probabilities before channel detail; show invalid channels with reasons; render per-channel visualizations without recomputing features or predictions.

- [ ] **Step 4: Verify GREEN and reopen the DOCX**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_reporting.py tests/test_pump_fault_app_export.py -q`

Expected: PASS; generated DOCX reopens and contains the required text and images.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/reporting pump_fault_app/export tests/test_pump_fault_app_reporting.py tests/test_pump_fault_app_export.py
git commit -m "feat: generate V3 multichannel diagnosis reports"
```

## Task 12: Upgrade the Streamlit pages and presentation adapters

**Files:**

- Modify: `pump_fault_app/ui/streamlit_app.py`
- Modify: `pump_fault_app/ui/pages/single_diagnosis.py`, `batch_diagnosis.py`, `report_view.py`, `history.py`, `system_overview.py`
- Modify: `pump_fault_app/presentation/*.py`, `pump_fault_app/ui/branding.py`
- Rewrite test: `tests/test_pump_fault_app_ui.py`

- [ ] **Step 1: Write failing navigation and content tests**

```python
def test_navigation_hides_model_optimization():
    labels = [item["title"] for item in build_navigation_items()]
    assert "模型持续优化" not in labels


def test_home_describes_only_v3_contract():
    sections = build_home_sections()
    text = repr(sections)
    for marker in ("43维", "CatBoost", "CH3", "CH4", "CH5", "4800", "2400", "5–5000"):
        assert marker in text
    assert "BP" not in text
    assert "21维" not in text
```

Add request-builder tests for CH3 only, CH3+CH4, and CH3+CH4+CH5 as three separately uploaded files with per-file columns and common sampling rate/rpm.

- [ ] **Step 2: Verify RED**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_ui.py -q`

Expected: FAIL because the existing UI shows BP/21-feature content and optimization navigation.

- [ ] **Step 3: Implement the V3 page hierarchy**

The single page must provide three optional channel upload sections, require at least one, prevent duplicate assignment, and enforce the all-or-none time-column rule before calling the service. The result page must show final fusion first, then a channel table and channel selector for four plots. Use “模型输出概率” and never “诊断置信度”.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_ui.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pump_fault_app/ui pump_fault_app/presentation tests/test_pump_fault_app_ui.py
git commit -m "feat: present CatBoost43 multichannel diagnosis UI"
```

## Task 13: Review and remove the obsolete BP application path

**Files:**

- Delete only after user confirmation: files and directories listed in the approved design.
- Modify: affected package `__init__.py` files and remaining tests.
- Create: `tests/test_pump_fault_app_no_legacy.py`

- [ ] **Step 1: Produce the exact deletion proposal without deleting**

Run:

```bash
git ls-files \
  pump_fault_app/feature_extraction/extractor.py \
  pump_fault_app/prediction/predictor.py \
  pump_fault_app/fusion/probability_fusion.py \
  pump_fault_app/inference/service.py \
  pump_fault_app/model_training \
  pump_fault_app/model_registry \
  pump_fault_app/sample_repository \
  pump_fault_app/ui/pages/model_optimization.py \
  tests/test_pump_fault_app_model_optimization.py
```

Present the resulting exact list to the user and wait for explicit deletion approval. Do not continue this task without that approval.

- [ ] **Step 2: Write a failing dead-reference scan test**

```python
def test_deployed_app_has_no_v2_bp_references():
    source = read_python_sources(Path("pump_fault_app"))
    for forbidden in ("FORMAL_V2_CONTRACT", "load_formal_bp_bundle", "StandardScaler", "MLPClassifier", "bp_bundle.joblib"):
        assert forbidden not in source
```

- [ ] **Step 3: Verify RED before deletion**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_no_legacy.py -q`

Expected: FAIL with the remaining obsolete references.

- [ ] **Step 4: Delete only the approved tracked paths and repair imports**

Use `git rm` with the exact user-approved list. Do not delete any file under `实验结果/`, any SQLite database, any thesis material, or any old design document. Update `__init__.py` exports so imports resolve exclusively to V3 modules.

- [ ] **Step 5: Verify GREEN and show Git deletions**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_no_legacy.py -q`

Run: `git diff --name-status --diff-filter=D HEAD^`

Expected: scan PASS and deletions exactly match the approved list.

- [ ] **Step 6: Commit**

```bash
git add pump_fault_app tests
git commit -m "refactor: remove obsolete BP application path"
```

## Task 14: Update operational documentation and run end-to-end acceptance

**Files:**

- Modify: `README.md`
- Modify: current operational `docs/pump_fault_app_*.md` files.
- Create/rewrite: `tests/test_pump_fault_app_end_to_end.py`

- [ ] **Step 1: Write failing end-to-end scenarios**

Parametrize real or deterministic fixture cases for CH3; CH4; CH3+CH4; CH3+CH4+CH5; one invalid channel; all invalid; one normal sample; and one fault sample. Each successful case must assert the same result object flows through inference, service, history, and Word report.

```python
@pytest.mark.parametrize("channels", [("CH3",), ("CH4",), ("CH3", "CH4"), ("CH3", "CH4", "CH5")])
def test_v3_end_to_end_channel_combinations(channels, fixture_factory, model_dir, tmp_path):
    app_result = run_single_diagnosis(fixture_factory(channels, model_dir, tmp_path))
    assert app_result.inference_result.status == "diagnosed"
    assert app_result.inference_result.valid_channels == channels
    assert sum(app_result.inference_result.fused_probabilities) == pytest.approx(1.0)
    assert app_result.history_record.contract_version == "formal-V3-catboost43"
    assert app_result.history_record.report_path.exists()
```

- [ ] **Step 2: Verify RED, then update docs and fixtures**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_pump_fault_app_end_to_end.py -q`

Expected: FAIL until every downstream path is V3-compatible.

Update current usage, architecture, acceptance, demo, walkthrough, defense, release, and screenshot documents. Do not rewrite historical files under `docs/superpowers/specs/` or `docs/superpowers/plans/`.

- [ ] **Step 3: Run targeted and full application suites**

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests/test_inference_contract.py tests/test_pump_fault_app_v3_models.py tests/test_pump_fault_app_feature_extraction.py tests/test_pump_fault_app_prediction.py tests/test_pump_fault_app_fusion.py tests/test_pump_fault_app_channel_fusion.py tests/test_pump_fault_app_multichannel_input.py tests/test_pump_fault_app_inference.py tests/test_pump_fault_app_services.py tests/test_pump_fault_app_batch.py tests/test_pump_fault_app_history.py tests/test_pump_fault_app_reporting.py tests/test_pump_fault_app_ui.py tests/test_pump_fault_app_end_to_end.py -q`

Run: `/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pytest tests -q`

Expected: all application and full tests PASS.

- [ ] **Step 4: Verify artifacts, source scans, and render output**

Run:

```bash
git diff --check
rg -n "FORMAL_V2_CONTRACT|load_formal_bp_bundle|bp_bundle.joblib|诊断置信度|21维|BP神经网络" pump_fault_app README.md docs/pump_fault_app_*.md
/Users/hewenhao/Documents/特征提取/.venv/bin/python -m pump_fault_app.app.self_check
```

Expected: no deployed-code or current-doc legacy matches; self-check verifies all three model hashes and a demo request. Start Streamlit from this worktree, verify all seven required input/result/history/report behaviors, and render/reopen the final Word report.

- [ ] **Step 5: Commit final acceptance changes**

```bash
git add README.md docs tests pump_fault_app
git commit -m "docs: complete CatBoost43 V3 acceptance"
```

- [ ] **Step 6: Final branch review without merging**

Run:

```bash
git status --short
git log --oneline --decorate 3bd3a3c..HEAD
git diff --stat 3bd3a3c..HEAD
git diff --name-status --diff-filter=D 3bd3a3c..HEAD
```

Expected: clean V3 branch, reviewable commits, and a deletion list containing only user-approved application files. Do not merge automatically.
