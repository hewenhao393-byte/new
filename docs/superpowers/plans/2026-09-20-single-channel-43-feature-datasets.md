# Single-Channel 43-Feature Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible CH3/CH4/CH5 raw 43-feature tables for record-level and 8 s/1 s/3 s temporal splits, without training models or overwriting existing artifacts.

**Architecture:** Add an isolated `feature_pipeline` Python package. It consumes the audited record manifest and measurement workbook, preprocesses each 12-second record once per channel, applies shared split definitions before windowing, extracts an identical ordered 43-feature vector, streams checkpoint files, and performs terminal leakage and feature-quality audits before publishing six CSV tables.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, PyWavelets, scikit-learn split utilities, openpyxl read-only workbook loading, pytest.

---

## File map

- Create `feature_pipeline/__init__.py`: package exports.
- Create `feature_pipeline/config.py`: immutable parameters, paths, and 43-feature order.
- Create `feature_pipeline/metadata.py`: audited manifest and workbook metadata loading.
- Create `feature_pipeline/preprocessing.py`: signal quality checks, resampling, and filtering.
- Create `feature_pipeline/splits.py`: stable record split and fixed temporal boundaries.
- Create `feature_pipeline/windowing.py`: record and block-bounded windows.
- Create `feature_pipeline/features.py`: ordered 43-feature extraction.
- Create `feature_pipeline/validation.py`: feature, alignment, and leakage checks.
- Create `feature_pipeline/pipeline.py`: checkpointed extraction and final CSV assembly.
- Create `feature_pipeline/main.py`: command-line interface.
- Create `tests/feature_pipeline/`: focused unit and integration tests.
- Create `实验结果/单通道43维_两种划分_20260920/`: new run output only; never replace existing incompatible content.

### Task 1: Lock the configuration and feature schema

**Files:**
- Create: `feature_pipeline/__init__.py`
- Create: `feature_pipeline/config.py`
- Test: `tests/feature_pipeline/test_config.py`

- [ ] **Step 1: Write the failing schema test**

```python
from feature_pipeline.config import CONFIG, FEATURE_NAMES

def test_formal_contract_is_fixed():
    assert CONFIG.original_fs == 20_000
    assert CONFIG.target_fs == 12_000
    assert CONFIG.window_size == 4_800
    assert CONFIG.step_size == 2_400
    assert CONFIG.temporal_train_samples == 96_000
    assert CONFIG.temporal_guard_samples == 12_000
    assert CONFIG.temporal_test_samples == 36_000
    assert CONFIG.split_mode_names == ("record", "temporal")
    assert len(FEATURE_NAMES) == 43
    assert len(set(FEATURE_NAMES)) == 43
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/feature_pipeline/test_config.py`

Expected: FAIL because `feature_pipeline.config` does not exist.

- [ ] **Step 3: Implement the immutable contract**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    original_fs: int = 20_000
    target_fs: int = 12_000
    bandpass_low: float = 5.0
    bandpass_high: float = 5_000.0
    window_size: int = 4_800
    step_size: int = 2_400
    order_half_width: float = 2.5
    wavelet: str = "db6"
    wavelet_level: int = 3
    envelope_low: float = 1_000.0
    envelope_high: float = 5_000.0
    envelope_spectrum_low: float = 5.0
    envelope_spectrum_high: float = 500.0
    eps: float = 1e-12
    random_seed: int = 2026
    temporal_train_samples: int = 96_000
    temporal_guard_samples: int = 12_000
    temporal_test_samples: int = 36_000
    train_block_samples: int = 19_200
    split_mode_names: tuple[str, str] = ("record", "temporal")

CONFIG = PipelineConfig()
FEATURE_NAMES = (
    "rms", "std", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "impulse_factor", "clearance_factor", "shape_factor",
    "rot_1x_energy_ratio", "rot_2x_energy_ratio", "rot_3x_energy_ratio",
    "rot_2x_1x_ratio", "rot_3x_1x_ratio", "harmonic_energy_ratio_1x_5x",
    "harmonic_energy_ratio_3x_5x", "rot_2x_harmonic_ratio",
    "rot_05x_1x_ratio", "noninteger_harmonic_energy_ratio",
    "spectral_entropy", "spectral_flatness", "spectral_centroid",
    "spectral_bandwidth", "band_energy_5_300_ratio",
    "band_energy_300_1000_ratio", "band_energy_1000_3000_ratio",
    "band_energy_3000_5000_ratio", "high_low_energy_ratio",
    "wp_energy_ratio_0", "wp_energy_ratio_1", "wp_energy_ratio_2",
    "wp_energy_ratio_3", "wp_energy_ratio_4", "wp_energy_ratio_5",
    "wp_energy_ratio_6", "wp_energy_ratio_7", "wp_energy_entropy",
    "env_kurtosis", "env_crest_factor", "env_spectral_entropy",
    "env_peak_energy_ratio", "env_peak_concentration", "env_peak_count",
)
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_config.py`

Expected: `1 passed`.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add feature_pipeline/__init__.py feature_pipeline/config.py tests/feature_pipeline/test_config.py
git commit -m "feat: define 43-feature dataset contract"
```

### Task 2: Load and audit paired record metadata

**Files:**
- Create: `feature_pipeline/metadata.py`
- Test: `tests/feature_pipeline/test_metadata.py`

- [ ] **Step 1: Write failing tests for identity and conservative severity handling**

```python
from feature_pipeline.metadata import make_record_id, merge_measurement_metadata

def test_record_id_uses_directory_and_column_not_csv_path():
    assert make_record_id("Motor-2", 100, "正常状态1", "0") == "Motor-2::100::正常状态1::0"

def test_unmatched_measurement_row_does_not_invent_severity(sample_manifest):
    merged, audit = merge_measurement_metadata(sample_manifest, [])
    assert merged.loc[0, "severity"] is None
    assert audit.loc[0, "status"] == "unmatched"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/feature_pipeline/test_metadata.py`

Expected: FAIL because metadata functions do not exist.

- [ ] **Step 3: Implement manifest validation and explicit workbook matching**

Implement these public interfaces:

```python
def make_record_id(device_id: str, speed_percent: int, state: str, column: str) -> str: ...
def load_record_manifest(path: Path) -> pd.DataFrame: ...
def load_measurement_rows(path: Path) -> list[dict[str, object]]: ...
def merge_measurement_metadata(records: pd.DataFrame, rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

`load_record_manifest` must require exactly one row per `group_id`, three existing channel paths, identical record columns, `eligible=True`, and six allowed labels. `load_measurement_rows` must open the workbook read-only and treat row 2 as headers. Matching must use normalized motor, speed, and source state/failure identity; zero or multiple matches must be audited and must not fabricate severity.

- [ ] **Step 4: Run focused metadata tests**

Run: `pytest -q tests/feature_pipeline/test_metadata.py`

Expected: all pass.

- [ ] **Step 5: Add a real-source read-only smoke test**

```python
def test_real_manifest_has_expected_record_contract(real_manifest_path):
    records = load_record_manifest(real_manifest_path)
    assert len(records) == 1845
    assert records.group_id.is_unique
    assert set(records.label) == {"正常", "转子不平衡", "联轴器不对中", "松动", "轴承故障", "汽蚀"}
```

Run: `pytest -q tests/feature_pipeline/test_metadata.py -k real_manifest`

Expected: pass without modifying source files.

- [ ] **Step 6: Commit Task 2**

```bash
git add feature_pipeline/metadata.py tests/feature_pipeline/test_metadata.py
git commit -m "feat: load audited record metadata"
```

### Task 3: Preprocess whole records and emit quality flags

**Files:**
- Create: `feature_pipeline/preprocessing.py`
- Test: `tests/feature_pipeline/test_preprocessing.py`

- [ ] **Step 1: Write failing behavioral tests**

```python
def test_preprocess_resamples_12_seconds_to_144000_points():
    t = np.arange(240_000) / 20_000
    raw = np.sin(2*np.pi*100*t) + 0.5
    y = preprocess_record(raw)
    assert y.shape == (144_000,)
    assert np.isfinite(y).all()

def test_impulsive_signal_is_flagged_not_deleted():
    x = np.zeros(240_000); x[1000] = 100
    quality = assess_quality(x)
    assert quality.quality_flag in {"warn", "pass"}
    assert "high_kurtosis" not in quality.rejection_reasons
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/feature_pipeline/test_preprocessing.py`

- [ ] **Step 3: Implement quality checks and fixed preprocessing**

Expose:

```python
@dataclass(frozen=True)
class QualityResult:
    quality_flag: str
    quality_reason: str
    rejection_reasons: tuple[str, ...]

def assess_quality(x: np.ndarray) -> QualityResult: ...
def preprocess_record(x: np.ndarray, config: PipelineConfig = CONFIG) -> np.ndarray: ...
```

Use `resample_poly(x - np.mean(x), 3, 5)`, `butter(4, [5, 5000], btype="bandpass", fs=12000, output="sos")`, and `sosfiltfilt`. Reject NaN/Inf, all-zero, and wrong-length temporal records; warn for long equal-value runs, clipping, or discontinuity indicators.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_preprocessing.py`

- [ ] **Step 5: Commit Task 3**

```bash
git add feature_pipeline/preprocessing.py tests/feature_pipeline/test_preprocessing.py
git commit -m "feat: preprocess complete vibration records"
```

### Task 4: Generate stable record and temporal splits before windowing

**Files:**
- Create: `feature_pipeline/splits.py`
- Create: `feature_pipeline/windowing.py`
- Test: `tests/feature_pipeline/test_splits.py`
- Test: `tests/feature_pipeline/test_windowing.py`

- [ ] **Step 1: Write failing split tests**

```python
def test_temporal_boundaries_are_exact_8_1_3_seconds():
    tail = temporal_bounds("record-a", 144_000, seed=2026, force_direction="test_tail")
    assert tail == {"train": (0, 96_000), "guard": (96_000, 108_000), "test": (108_000, 144_000)}
    head = temporal_bounds("record-b", 144_000, seed=2026, force_direction="test_head")
    assert head == {"test": (0, 36_000), "guard": (36_000, 48_000), "train": (48_000, 144_000)}

def test_record_split_never_splits_a_record(sample_records):
    assignment = make_record_split(sample_records)
    assert assignment.record_id.is_unique
    assert set(assignment.split) <= {"train_dev", "test"}
```

- [ ] **Step 2: Write failing window-boundary tests**

```python
def test_temporal_windows_never_touch_guard_or_cross_blocks():
    windows = list(iter_temporal_windows(144_000, temporal_bounds("a", 144_000, force_direction="test_tail")))
    assert all(not (w.start < 108_000 and w.end > 96_000) for w in windows)
    assert all(w.split != "train" or w.start // 19_200 == (w.end - 1) // 19_200 for w in windows)
```

- [ ] **Step 3: Run and verify RED**

Run: `pytest -q tests/feature_pipeline/test_splits.py tests/feature_pipeline/test_windowing.py`

- [ ] **Step 4: Implement deterministic splits and bounded windows**

`make_record_split` must stratify as closely as feasible by label/device/speed/state while keeping one row per record and seed 2026. Existing split files must be validated and reused. `temporal_bounds` must require exactly 144000 samples and choose direction from a stable SHA-256 hash of `(seed, record_id)`, not Python's process-randomized `hash()`.

`iter_temporal_windows` must first divide the 96000-sample train region into 19200-sample blocks, then slide independently inside each block. It must slide independently inside the 36000-sample test region and never emit guard windows.

- [ ] **Step 5: Run and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_splits.py tests/feature_pipeline/test_windowing.py`

- [ ] **Step 6: Commit Task 4**

```bash
git add feature_pipeline/splits.py feature_pipeline/windowing.py tests/feature_pipeline/test_splits.py tests/feature_pipeline/test_windowing.py
git commit -m "feat: add leakage-safe record and temporal splits"
```

### Task 5: Extract the exact ordered 43 features

**Files:**
- Create: `feature_pipeline/features.py`
- Test: `tests/feature_pipeline/test_features.py`

- [ ] **Step 1: Write failing invariants tests**

```python
def test_extracts_exact_ordered_finite_43_features():
    t = np.arange(4_800) / 12_000
    x = np.sin(2*np.pi*25*t) + 0.2*np.sin(2*np.pi*50*t)
    values = extract_features(x, rpm=1500)
    assert tuple(values) == FEATURE_NAMES
    assert len(values) == 43
    assert np.isfinite(list(values.values())).all()

def test_order_band_clips_low_edge_and_rejects_outside_center():
    assert order_band(6.17) == (5.0, 8.67)
    with pytest.raises(OutOfBandOrderError):
        order_band(4.99)

def test_partitioned_band_ratios_sum_to_one():
    values = extract_features(np.random.default_rng(2026).normal(size=4_800), rpm=740)
    total = sum(values[name] for name in BAND_RATIO_NAMES)
    assert abs(total - 1.0) < 1e-6

def test_wavelet_packet_ratios_sum_to_one():
    values = extract_features(np.random.default_rng(7).normal(size=4_800), rpm=1480)
    total = sum(values[f"wp_energy_ratio_{i}"] for i in range(8))
    assert abs(total - 1.0) < 1e-8
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/feature_pipeline/test_features.py`

- [ ] **Step 3: Implement one shared spectrum and focused feature helpers**

Implement `time_features`, `order_features`, `spectral_features`, `band_features`, `wavelet_packet_features`, `envelope_features`, and `extract_features`. Use Pearson kurtosis (`fisher=False`), Hann-windowed `rfft`, 5–5000 Hz analysis masks, clipped ±2.5 Hz order bands, half-open energy partitions, `db6` level 3 natural-order terminal nodes, and envelope peak detection with 5% maximum prominence, 5 Hz minimum spacing, and at most five strongest peaks.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_features.py`

- [ ] **Step 5: Add independent hand-calculation tests for RMS, shape factor, and a pure-tone 1X band**

Run: `pytest -q tests/feature_pipeline/test_features.py`

Expected: all feature tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add feature_pipeline/features.py tests/feature_pipeline/test_features.py
git commit -m "feat: extract ordered 43-feature vectors"
```

### Task 6: Validate leakage, alignment, and final tables

**Files:**
- Create: `feature_pipeline/validation.py`
- Test: `tests/feature_pipeline/test_validation.py`

- [ ] **Step 1: Write failing rejection tests**

```python
def test_rejects_record_leakage():
    assignments = pd.DataFrame({"record_id": ["a", "a"], "split": ["train_dev", "test"]})
    with pytest.raises(ValidationError, match="record leakage"):
        validate_record_split(assignments)

def test_rejects_misaligned_channels(ch3, ch4, ch5):
    ch5 = ch5.iloc[:-1]
    with pytest.raises(ValidationError, match="channel alignment"):
        validate_channel_alignment({3: ch3, 4: ch4, 5: ch5})
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/feature_pipeline/test_validation.py`

- [ ] **Step 3: Implement terminal validators**

Validate the exact feature columns, finite values, unique window keys, channel-key equality, label/RPM equality, record disjointness, exact 12000-sample guard, no interval intersection, no train-block crossing, band-ratio tolerance `1e-6`, wavelet-ratio tolerance, and output row counts. Write machine-readable `leakage_audit.json` only from independently recomputed checks.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_validation.py`

- [ ] **Step 5: Commit Task 6**

```bash
git add feature_pipeline/validation.py tests/feature_pipeline/test_validation.py
git commit -m "feat: validate feature tables and split leakage"
```

### Task 7: Build checkpointed orchestration and CLI

**Files:**
- Create: `feature_pipeline/pipeline.py`
- Create: `feature_pipeline/main.py`
- Test: `tests/feature_pipeline/test_pipeline_integration.py`

- [ ] **Step 1: Write a failing two-record integration test**

```python
def test_two_record_pipeline_writes_and_reopens_all_outputs(tmp_path, two_record_manifest, workbook_path):
    run_pipeline(two_record_manifest, workbook_path, tmp_path, workers=1)
    for folder in ("file_split", "temporal_split"):
        base = tmp_path / folder
        tables = {channel: pd.read_csv(base / f"features_ch{channel}.csv") for channel in (3, 4, 5)}
        validate_channel_alignment(tables)
        assert (base / "dataset_summary.csv").exists()
        assert (base / "quality_audit.csv").exists()
        assert json.loads((base / "leakage_audit.json").read_text())["passed"] is True
    assert (tmp_path / "file_split" / "file_split.csv").exists()
    assert (tmp_path / "temporal_split" / "temporal_split.csv").exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/feature_pipeline/test_pipeline_integration.py`

- [ ] **Step 3: Implement checkpointed extraction**

Expose:

```python
def run_pipeline(manifest_path: Path, workbook_path: Path, output_root: Path,
                 modes: tuple[str, ...] = ("record", "temporal"),
                 limit: int | None = None, workers: int = 1) -> None: ...
```

Read each channel CSV once per file group, extract the requested record columns, preprocess each complete record, and write per-record/channel checkpoint CSV or Parquet files under a newly created run checkpoint directory. Final assembly must sort by `(record_id, split, start_sample, channel)`. If a final output directory exists with a different configuration or input hash, stop. Never delete or silently replace it.

- [ ] **Step 4: Add CLI arguments**

```text
python -m feature_pipeline.main \
  --manifest <record_pairs.csv> \
  --workbook <测量工况总表.xlsx> \
  --output <new-output-directory> \
  --mode both --workers 3 [--limit N]
```

- [ ] **Step 5: Run integration test and verify GREEN**

Run: `pytest -q tests/feature_pipeline/test_pipeline_integration.py`

- [ ] **Step 6: Run the complete test suite for the new package**

Run: `pytest -q tests/feature_pipeline`

Expected: zero failures and zero unexpected warnings.

- [ ] **Step 7: Commit Task 7**

```bash
git add feature_pipeline/pipeline.py feature_pipeline/main.py tests/feature_pipeline/test_pipeline_integration.py
git commit -m "feat: orchestrate checkpointed feature datasets"
```

### Task 8: Limited real-data run and acceptance

**Files:**
- Create: `实验结果/单通道43维_两种划分_20260920/` outputs only after checks pass.

- [ ] **Step 1: Run six representative records across all labels where available**

Run the CLI with `--limit` or an explicit test manifest. Inspect logs for preprocessing failures, metadata ambiguity, invalid orders, and non-finite features.

- [ ] **Step 2: Reopen and validate all limited-run outputs**

Run an independent validation command that reads the saved CSV files rather than in-memory frames. Expected: all alignment, leakage, 43-feature, band-ratio, and wavelet-ratio checks pass.

- [ ] **Step 3: Estimate full-run storage and duration from measured limited-run throughput**

Record the estimate in `run_manifest.json`; do not infer it before measuring.

- [ ] **Step 4: Run all 1845 records only after the limited run passes**

Use the fixed manifest, workbook, seed, and fresh output directory. Do not modify source CSVs, the workbook, old feature tables, or old models.

- [ ] **Step 5: Perform fresh final verification**

Run:

```bash
pytest -q tests/feature_pipeline
python -m feature_pipeline.main --verify-only --output '实验结果/单通道43维_两种划分_20260920'
```

Expected: tests report zero failures; verification reports six readable feature tables, two reusable split files, no leakage, no NaN/Inf, exact channel alignment, valid 43-feature order, band-ratio sum tolerance below `1e-6`, and input/config/split hashes.

- [ ] **Step 6: Review git status and commit only source/tests/docs**

Do not commit large generated feature tables unless the user explicitly requests it.
