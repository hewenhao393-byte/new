# Channel 4 Six-Class Pump Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a full Python diagnosis pipeline that uses only channel-4 vibration CSVs to produce auditable six-class train/test splits, features, grouped model selection, final evaluation, and deliverables under `outputs/channel4_six_class_v1/`.

**Architecture:** Rebuild the project around a small `pump_diagnosis` package with explicit stage boundaries and file-based artifacts between stages. Keep training/test isolation strict by making the source CSV the atomic split unit, fitting every learned transformation on training data only, and using grouped cross-validation before a single final test evaluation.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, scikit-learn, pywavelets, matplotlib, seaborn, joblib, openpyxl, pytest

---

## File Structure

- `config.py`
  - Replace the current partial configuration with one authoritative home for all paths, sampling, filtering, feature-selection, CV, and model-search parameters.
- `pump_diagnosis/__init__.py`
  - Package marker and minimal version export.
- `pump_diagnosis/labels.py`
  - Approved six-class mapping, class order, excluded prefixes, and label lookup helpers.
- `pump_diagnosis/metadata.py`
  - Raw file discovery, workbook parsing, signal-column inventory, duplicate hashing, and grouped train/test splitting.
- `pump_diagnosis/signal_processing.py`
  - DC removal, `resample_poly(3, 5)`, Butterworth bandpass, window iteration, and preprocessing quality summaries.
- `pump_diagnosis/features.py`
  - Deterministic feature list, feature dictionary rows, and feature extraction from one processed window.
- `pump_diagnosis/pipeline.py`
  - Stage-1 feature pipeline that reads indexed runs, processes them source-file by source-file, writes feature tables, and emits audit artifacts.
- `pump_diagnosis/modeling.py`
  - Training-only feature filtering, leakage audit, grouped CV helpers, and feature-selection summaries.
- `pump_diagnosis/model_training.py`
  - SVM, RandomForest, and MLP tuning/training/final metrics/persistence.
- `pump_diagnosis/modeling_runner.py`
  - End-to-end modeling stage driver from raw feature CSVs to final reports.
- `pump_diagnosis/plots.py`
  - Required PNG plots for data distribution, filter response, preprocessing spectra, wavelet energy, confusion matrices, and model comparison.
- `pump_diagnosis/runner.py`
  - CLI stage orchestration: `inspect`, `split`, `preprocess`, `features`, `select`, `train`, `evaluate`, `all`.
- `train_models.py`
  - Replace the current standalone training script with a thin compatibility shim that forwards to `pump_diagnosis.runner`.
- `requirements.txt`
  - Runtime dependencies pinned to major/minor ranges that are available in this environment.
- `README.md`
  - Exact commands, feature formulas, output tree, data assumptions, and one-pass reproduction notes.
- `tests/test_rebuild_metadata.py`
  - Metadata indexing, workbook mapping, and split-leakage tests.
- `tests/test_rebuild_signal_features.py`
  - Preprocessing, windowing, and feature finite-value tests.
- `tests/test_rebuild_pipeline.py`
  - Stage-1 pipeline outputs and feature dictionary tests.
- `tests/test_grouped_modeling.py`
  - Training-only filtering, grouped CV, model outputs, and full modeling run tests.
- `tests/test_rebuild_outputs.py`
  - Config serialization, lazy imports, plotting outputs, and feature CSV validation tests.

## Task 1: Rebuild the Package Skeleton and Configuration

**Files:**
- Create: `pump_diagnosis/__init__.py`
- Create: `pump_diagnosis/labels.py`
- Modify: `config.py`
- Create: `requirements.txt`
- Test: `tests/test_rebuild_outputs.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from config import PipelineConfig, ModelingConfig


def test_pipeline_config_uses_channel4_output_contract(tmp_path):
    config = PipelineConfig(output_root=tmp_path / "outputs")
    assert config.channel == 4
    assert config.window_size == 4096
    assert config.step_size == 2048
    assert config.output_root.name == "outputs"
    assert config.label_order == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    )


def test_modeling_config_matches_grouped_selection_defaults():
    config = ModelingConfig()
    assert config.correlation_threshold == 0.90
    assert config.cv_folds == 5
    assert config.mlp_hidden_layer_sizes == (64, 32)
```

- [ ] **Step 2: Run the test to confirm the current config is wrong**

Run: `pytest tests/test_rebuild_outputs.py::test_pipeline_config_uses_channel4_output_contract tests/test_rebuild_outputs.py::test_modeling_config_matches_grouped_selection_defaults -v`

Expected: FAIL because `window_size`, `step_size`, `output_root`, and `correlation_threshold` still match the old partial implementation.

- [ ] **Step 3: Implement the package skeleton and new config values**

```python
# pump_diagnosis/__init__.py
__all__ = ["__version__"]
__version__ = "0.1.0"
```

```python
# pump_diagnosis/labels.py
LABEL_ORDER = (
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
)

PREFIX_TO_LABEL = {
    "正常状态": "正常",
    "泵不平衡": "转子不平衡",
    "电机不平衡": "转子不平衡",
    "角向不对中": "联轴器不对中",
    "平行不对中": "联轴器不对中",
    "组合不对中": "联轴器不对中",
    "软脚": "松动",
    "电机地脚松动": "松动",
    "泵地脚松动": "松动",
    "轴承内圈故障": "轴承故障",
    "轴承外圈故障": "轴承故障",
    "轴承滚动体故障": "轴承故障",
    "轴承污染": "轴承故障",
    "泵轴承故障": "轴承故障",
    "吸入口汽蚀": "汽蚀",
    "出口汽蚀": "汽蚀",
}
```

```python
# config.py
@dataclass(frozen=True)
class PipelineConfig:
    output_root: Path = field(default_factory=lambda: Path("outputs") / "channel4_six_class_v1")
    channel: int = 4
    original_fs: int = 20_000
    processed_fs: int = 12_000
    window_size: int = 4096
    step_size: int = 2048
    missing_feature_threshold: float = 0.20
    correlation_threshold: float = 0.90
    near_zero_variance_threshold: float = 1e-12
    label_order: tuple[str, ...] = LABEL_ORDER


@dataclass(frozen=True)
class ModelingConfig:
    feature_root: Path = field(default_factory=lambda: Path("outputs") / "channel4_six_class_v1")
    output_root: Path = field(default_factory=lambda: Path("outputs") / "channel4_six_class_v1" / "modeling")
    correlation_threshold: float = 0.90
    cv_folds: int = 5
    mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32)
```

```text
# requirements.txt
numpy>=2.0,<3.0
pandas>=2.2,<3.0
scipy>=1.14,<2.0
scikit-learn>=1.6,<2.0
PyWavelets>=1.8,<2.0
matplotlib>=3.10,<4.0
seaborn>=0.13,<1.0
joblib>=1.4,<2.0
openpyxl>=3.1,<4.0
pytest>=9.0,<10.0
```

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_rebuild_outputs.py::test_pipeline_config_uses_channel4_output_contract tests/test_rebuild_outputs.py::test_modeling_config_matches_grouped_selection_defaults -v`

Expected: PASS

- [ ] **Step 5: Commit the configuration baseline**

```bash
git add config.py pump_diagnosis/__init__.py pump_diagnosis/labels.py requirements.txt tests/test_rebuild_outputs.py
git commit -m "feat: define channel4 configuration baseline"
```

## Task 2: Build Metadata Indexing, Workbook Mapping, and Source-File Splitting

**Files:**
- Create: `pump_diagnosis/metadata.py`
- Modify: `tests/test_rebuild_metadata.py`
- Modify: `tests/test_rebuild_pipeline.py`

- [ ] **Step 1: Write the failing metadata tests**

```python
def test_build_file_index_discovers_only_channel4_and_target_labels(tmp_path):
    index = build_file_index(PipelineConfig(data_root=tmp_path, output_root=tmp_path / "output"))
    assert set(index["channel"]) == {4}
    assert set(index["label"]).issubset(set(PipelineConfig().label_order))


def test_split_file_index_preserves_source_file_groups_and_labels(tmp_path):
    train, test, report = split_file_index(file_index, PipelineConfig(output_root=tmp_path / "output"))
    assert report["source_file_overlap_count"] == 0
    assert report["hash_overlap_count"] == 0
    assert sorted(train["label"].unique()) == sorted(test["label"].unique())
```

- [ ] **Step 2: Run the metadata tests to capture the missing module failure**

Run: `pytest tests/test_rebuild_metadata.py tests/test_rebuild_pipeline.py::test_condition_catalog_maps_chinese_faults_to_actual_rpm -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pump_diagnosis'` or missing `build_file_index` / `split_file_index`.

- [ ] **Step 3: Implement workbook parsing, file inventory, hashing, and grouped splitting**

```python
# pump_diagnosis/metadata.py
@dataclass(frozen=True)
class ConditionCatalog:
    rpm_by_key: dict[tuple[str, int, str], float]

    @classmethod
    def from_workbook(cls, workbook_path: Path) -> "ConditionCatalog":
        frame = pd.read_excel(workbook_path, sheet_name="Ordered Measurements", header=1)
        frame = frame.rename(
            columns={
                "Setup": "setup",
                "Failure description": "failure_description",
                "Speed (%)": "speed_percent",
                "Speed (RPM)": "speed_rpm",
            }
        )
        rpm_by_key: dict[tuple[str, int, str], float] = {}
        for row in frame[["setup", "failure_description", "speed_percent", "speed_rpm"]].dropna().itertuples(index=False):
            key = (_normalize_machine_id(row.setup), int(round(float(row.speed_percent) * 100)), _normalize_fault_name(row.failure_description))
            rpm = float(row.speed_rpm)
            if key in rpm_by_key and not np.isclose(rpm_by_key[key], rpm):
                raise ValueError(f"冲突转速: {key}")
            rpm_by_key[key] = rpm
        return cls(rpm_by_key=rpm_by_key)

    def rpm_for(self, machine_id: str, speed_percent: int, raw_fault: str) -> float | None:
        return self.rpm_by_key.get((machine_id, speed_percent, raw_fault))


def build_file_index(config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(config.data_root.rglob("*通道4.csv")):
        raw_fault = csv_path.parent.name
        label = map_fault_to_label(raw_fault)
        if label is None:
            continue
        frame = pd.read_csv(csv_path)
        signal_columns = [column for column in frame.columns if column != "time"]
        file_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        for signal_column in signal_columns:
            rows.append(
                {
                    "file_path": str(csv_path),
                    "label": label,
                    "raw_fault": raw_fault,
                    "source_column": signal_column,
                    "run_id": f"{csv_path.stem}_{signal_column}",
                    "sha256": file_hash,
                    "channel": 4,
                    "n_samples": int(frame[signal_column].shape[0]),
                }
            )
    return pd.DataFrame(rows)


def split_file_index(index: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    grouped = index[["file_path", "label", "sha256"]].drop_duplicates()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    train_idx, test_idx = next(splitter.split(grouped["file_path"], grouped["label"]))
    train_files = set(grouped.iloc[train_idx]["file_path"])
    test_files = set(grouped.iloc[test_idx]["file_path"])
    train_rows = index[index["file_path"].isin(train_files)].reset_index(drop=True)
    test_rows = index[index["file_path"].isin(test_files)].reset_index(drop=True)
    leakage_report = {
        "source_file_overlap_count": len(train_files.intersection(test_files)),
        "hash_overlap_count": len(set(train_rows["sha256"]).intersection(test_rows["sha256"])),
    }
    return train_rows, test_rows, leakage_report
```

- [ ] **Step 4: Run the metadata test file**

Run: `pytest tests/test_rebuild_metadata.py tests/test_rebuild_pipeline.py::test_condition_catalog_maps_chinese_faults_to_actual_rpm tests/test_rebuild_pipeline.py::test_condition_catalog_rejects_conflicting_rpm_rows -v`

Expected: PASS

- [ ] **Step 5: Commit the metadata stage**

```bash
git add pump_diagnosis/metadata.py tests/test_rebuild_metadata.py tests/test_rebuild_pipeline.py
git commit -m "feat: index channel4 files and split source groups"
```

## Task 3: Implement Preprocessing, Windowing, and Multi-Domain Features

**Files:**
- Create: `pump_diagnosis/signal_processing.py`
- Create: `pump_diagnosis/features.py`
- Modify: `tests/test_rebuild_signal_features.py`

- [ ] **Step 1: Write the failing preprocessing and feature tests**

```python
def test_preprocess_run_resamples_and_filters_signal():
    processed, summary = preprocess_run(raw_signal, rpm=2070.0, config=PipelineConfig())
    assert processed.fs == 12_000
    assert abs(summary["mean_after_dc"]) < 1e-10
    assert processed.samples.shape[0] == 12_000


def test_extract_window_features_returns_expected_schema():
    features = extract_window_features(window, rpm=2070.0, config=PipelineConfig())
    assert list(features) == FEATURE_COLUMNS
    assert np.isfinite(pd.Series(features).to_numpy()).all()
```

- [ ] **Step 2: Run the preprocessing test file**

Run: `pytest tests/test_rebuild_signal_features.py -v`

Expected: FAIL because preprocessing and feature functions do not exist yet.

- [ ] **Step 3: Implement DC removal, resampling, filtering, windows, and feature extraction**

```python
# pump_diagnosis/signal_processing.py
def preprocess_run(samples: np.ndarray, rpm: float | None, config: PipelineConfig) -> ProcessedRun:
    centered = samples - np.mean(samples)
    resampled = signal.resample_poly(centered, up=config.resample_up, down=config.resample_down)
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.bandpass_low, config.bandpass_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, resampled)
    return ProcessedRun(samples=filtered.astype(np.float64), fs=config.processed_fs, rpm=rpm)


def iter_windows(samples: np.ndarray, window_size: int, step_size: int) -> Iterator[tuple[int, np.ndarray]]:
    for start in range(0, samples.shape[0] - window_size + 1, step_size):
        yield start, samples[start : start + window_size]
```

```python
# pump_diagnosis/features.py
TIME_FEATURES = ["mean", "std", "rms", "peak", "peak_to_peak", "skewness", "kurtosis", "crest_factor", "shape_factor", "impulse_factor", "margin_factor", "clearance_factor", "variance", "energy", "entropy", "zero_crossing_rate", "waveform_length", "median_abs"]
FREQUENCY_FEATURES = ["freq_centroid", "freq_rms", "freq_std", "freq_skewness", "freq_kurtosis", "spectral_entropy", "peak_frequency", "peak_amplitude", "band_energy_10_500", "band_energy_500_1000", "band_energy_1000_2000", "band_energy_2000_5000", "band_ratio_10_500", "band_ratio_500_1000", "band_ratio_1000_2000", "band_ratio_2000_5000", "spec_flatness", "spec_rolloff_85", "spec_rolloff_95", "spec_flux"]
ROTATION_FEATURES = ["amp_1x", "amp_2x", "amp_3x", "energy_1x", "energy_2x", "energy_3x", "ratio_2x_to_1x", "ratio_3x_to_1x", "rotation_band_energy"]
WAVELET_FEATURES = ["wp_energy_aaa", "wp_energy_aad", "wp_energy_ada", "wp_energy_add", "wp_energy_daa", "wp_energy_dad", "wp_energy_dda", "wp_energy_ddd", "wp_entropy_aaa", "wp_entropy_aad", "wp_entropy_ada", "wp_entropy_add", "wp_entropy_daa", "wp_entropy_dad", "wp_entropy_dda", "wp_entropy_ddd", "wp_total_entropy"]
ENVELOPE_FEATURES = ["env_rms", "env_peak", "env_kurtosis", "env_entropy", "env_peak_frequency", "env_peak_amplitude", "env_band_energy_0_100", "env_band_energy_100_500", "env_band_energy_500_1000", "env_band_energy_1000_3000", "env_ratio_0_100", "env_ratio_100_500", "env_ratio_500_1000", "env_ratio_1000_3000", "env_amp_1x", "env_amp_2x", "env_ratio_2x_to_1x", "env_spectral_entropy", "env_spec_centroid", "env_spec_rms"]

FEATURE_COLUMNS = TIME_FEATURES + FREQUENCY_FEATURES + ROTATION_FEATURES + WAVELET_FEATURES + ENVELOPE_FEATURES


def extract_window_features(window: np.ndarray, rpm: float | None, config: PipelineConfig) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(compute_time_features(window))
    features.update(compute_frequency_features(window, config.processed_fs))
    features.update(compute_rotation_features(window, rpm, config))
    features.update(compute_wavelet_packet_features(window, config.wavelet, config.wavelet_level))
    features.update(compute_envelope_features(window, rpm, config))
    ordered = {name: float(features[name]) for name in FEATURE_COLUMNS}
    if not np.isfinite(np.fromiter(ordered.values(), dtype=np.float64)).all():
        raise ValueError("non-finite feature detected")
    return ordered
```

- [ ] **Step 4: Run the signal and feature tests**

Run: `pytest tests/test_rebuild_signal_features.py -v`

Expected: PASS

- [ ] **Step 5: Commit the deterministic feature layer**

```bash
git add pump_diagnosis/signal_processing.py pump_diagnosis/features.py tests/test_rebuild_signal_features.py
git commit -m "feat: add preprocessing and vibration feature extraction"
```

## Task 4: Build the Stage-1 Feature Pipeline and Auditable Outputs

**Files:**
- Create: `pump_diagnosis/pipeline.py`
- Create: `pump_diagnosis/plots.py`
- Modify: `tests/test_rebuild_pipeline.py`
- Modify: `tests/test_rebuild_outputs.py`

- [ ] **Step 1: Write the failing stage-1 pipeline tests**

```python
def test_extract_split_features_skips_nonfinite_raw_runs_and_reuses_outputs(tmp_path):
    result = extract_split_features(manifest, "train", PipelineConfig(output_root=tmp_path / "output"))
    assert result.feature_csv.exists()
    assert result.quality_csv.exists()
    assert result.summary_json.exists()


def test_generate_stage1_plots_creates_four_nonempty_images(tmp_path):
    paths = generate_stage1_plots(train_manifest, test_manifest, feature_csv, PipelineConfig(output_root=tmp_path / "output"))
    assert set(paths) == {"class_counts", "filter_response", "preprocessing_spectra", "wavelet_energy"}
```

- [ ] **Step 2: Run the stage-1 tests**

Run: `pytest tests/test_rebuild_pipeline.py tests/test_rebuild_outputs.py -v`

Expected: FAIL because `extract_split_features`, `validate_feature_csv`, and plotting helpers are incomplete.

- [ ] **Step 3: Implement the source-file streaming pipeline and output validators**

```python
# pump_diagnosis/pipeline.py
METADATA_COLUMNS = [
    "sample_id",
    "label",
    "run_id",
    "file_path",
    "machine_id",
    "condition_id",
    "rpm",
    "channel",
    "window_index",
    "window_start",
    "window_end",
    "original_fs",
    "processed_fs",
]


def extract_split_features(manifest: pd.DataFrame, split_name: str, config: PipelineConfig) -> Stage1Result:
    feature_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for file_path, file_rows in manifest.groupby("file_path", sort=True):
        raw = pd.read_csv(file_path)
        for row in file_rows.itertuples(index=False):
            samples = raw[row.source_column].to_numpy(dtype=np.float64)
            if not np.isfinite(samples).all():
                quality_rows.append({"run_id": row.run_id, "reason": "raw_nonfinite"})
                continue
            processed = preprocess_run(samples, getattr(row, "rpm", None), config)
            for window_index, (start, window) in enumerate(iter_windows(processed.samples, config.window_size, config.step_size)):
                feature_row = {column: getattr(row, column) for column in ["label", "run_id", "file_path", "machine_id", "condition_id", "rpm", "channel"]}
                feature_row.update(
                    {
                        "sample_id": f"{row.run_id}_window_{window_index}",
                        "window_index": window_index,
                        "window_start": start / config.processed_fs,
                        "window_end": (start + config.window_size) / config.processed_fs,
                        "original_fs": config.original_fs,
                        "processed_fs": config.processed_fs,
                    }
                )
                feature_row.update(extract_window_features(window, getattr(row, "rpm", None), config))
                feature_rows.append(feature_row)
    features = pd.DataFrame(feature_rows, columns=METADATA_COLUMNS + FEATURE_COLUMNS)
    quality = pd.DataFrame(quality_rows)
    feature_csv = config.output_root / f"{split_name}_features_raw.csv"
    quality_csv = config.output_root / f"{split_name}_quality_report.csv"
    summary_json = config.output_root / f"{split_name}_feature_summary.json"
    features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    quality.to_csv(quality_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps({"rows": len(features), "skipped_runs": len(quality)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return Stage1Result(feature_csv=feature_csv, quality_csv=quality_csv, summary_json=summary_json)


def write_feature_dictionary(path: Path) -> None:
    rows = build_feature_dictionary_rows()
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
```

```python
# pump_diagnosis/plots.py
def generate_stage1_plots(train_manifest: pd.DataFrame, test_manifest: pd.DataFrame, feature_csv: Path, config: PipelineConfig) -> dict[str, Path]:
    output_dir = config.output_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "class_counts": output_dir / "class_counts.png",
        "filter_response": output_dir / "filter_response.png",
        "preprocessing_spectra": output_dir / "preprocessing_spectra.png",
        "wavelet_energy": output_dir / "wavelet_energy.png",
    }
    plot_class_counts(train_manifest, test_manifest, paths["class_counts"], config.plot_dpi)
    plot_filter_response(paths["filter_response"], config)
    plot_preprocessing_spectra(feature_csv, paths["preprocessing_spectra"], config.plot_dpi)
    plot_wavelet_energy(feature_csv, paths["wavelet_energy"], config.plot_dpi)
    return paths
```

```python
# pump_diagnosis/runner.py
def validate_feature_csv(path: Path, config: PipelineConfig) -> dict[str, object]:
    frame = pd.read_csv(path)
    if list(frame.columns) != METADATA_COLUMNS + FEATURE_COLUMNS:
        raise ValueError("特征表字段顺序不正确")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("特征表存在NaN或Inf")
    return {"rows": len(frame), "feature_count": len(FEATURE_COLUMNS), "labels": frame["label"].value_counts().to_dict()}
```

- [ ] **Step 4: Run the stage-1 regression tests**

Run: `pytest tests/test_rebuild_pipeline.py tests/test_rebuild_outputs.py -v`

Expected: PASS

- [ ] **Step 5: Commit the feature-generation stage**

```bash
git add pump_diagnosis/pipeline.py pump_diagnosis/plots.py pump_diagnosis/runner.py tests/test_rebuild_pipeline.py tests/test_rebuild_outputs.py
git commit -m "feat: build auditable channel4 feature pipeline"
```

## Task 5: Implement Training-Only Filtering, Grouped CV, and Final Models

**Files:**
- Create: `pump_diagnosis/modeling.py`
- Create: `pump_diagnosis/model_training.py`
- Create: `pump_diagnosis/modeling_runner.py`
- Modify: `tests/test_grouped_modeling.py`

- [ ] **Step 1: Write the failing modeling tests**

```python
def test_training_filter_keeps_rms_and_ignores_test_variance():
    result = fit_training_feature_filter(train, feature_columns, ModelingConfig())
    assert "rms" in result.kept_features
    assert "std" not in result.kept_features
    assert result.removed_reasons["near_zero_feature"] == "near_zero_variance"


def test_top_k_selection_uses_file_grouped_folds_without_overlap(tmp_path):
    result = evaluate_top_k_grouped(frame, feature_columns, ModelingConfig(output_root=tmp_path, top_k_candidates=(2, 3)))
    assert result.fold_results["group_overlap_count"].eq(0).all()
```

- [ ] **Step 2: Run the modeling tests**

Run: `pytest tests/test_grouped_modeling.py -v`

Expected: FAIL because grouped modeling helpers and final training runner do not exist yet.

- [ ] **Step 3: Implement leakage audit, feature selection, grouped CV, and three-model training**

```python
# pump_diagnosis/modeling.py
class LeakageError(RuntimeError):
    def __init__(self, report: dict[str, int]):
        super().__init__("training/test leakage detected")
        self.report = report


def audit_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    report = {
        "file_path_overlap_count": len(set(train["file_path"]).intersection(test["file_path"])),
        "run_id_overlap_count": len(set(train["run_id"]).intersection(test["run_id"])),
        "sample_id_overlap_count": len(set(train["sample_id"]).intersection(test["sample_id"])),
    }
    if any(report.values()):
        raise LeakageError(report)
    return report
```

```python
# pump_diagnosis/model_training.py
def train_and_evaluate_models(train: pd.DataFrame, test: pd.DataFrame, selected_features: list[str], config: ModelingConfig) -> pd.DataFrame:
    rf = RandomForestClassifier(n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth, min_samples_split=config.rf_min_samples_split, min_samples_leaf=config.rf_min_samples_leaf, max_features=config.rf_max_features, bootstrap=config.rf_bootstrap, class_weight=config.rf_class_weight, random_state=config.random_state, n_jobs=config.rf_n_jobs)
    svm = Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel=config.svm_kernel, C=config.svm_c, gamma=config.svm_gamma, class_weight=config.svm_class_weight, probability=False, random_state=config.random_state))])
    mlp = Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=config.mlp_hidden_layer_sizes, activation=config.mlp_activation, solver=config.mlp_solver, alpha=config.mlp_alpha, learning_rate_init=config.mlp_learning_rate_init, max_iter=config.mlp_max_iter, early_stopping=config.mlp_early_stopping, validation_fraction=config.mlp_validation_fraction, n_iter_no_change=config.mlp_n_iter_no_change, batch_size=config.mlp_batch_size, random_state=config.random_state))])
    metrics_rows = [fit_one_model("RandomForest", rf, train, test, selected_features, config), fit_one_model("SVM", svm, train, test, selected_features, config), fit_one_model("MLP", mlp, train, test, selected_features, config)]
    return pd.DataFrame(metrics_rows)
```

```python
# pump_diagnosis/modeling_runner.py
def run_full_modeling(config: ModelingConfig) -> dict[str, object]:
    train = pd.read_csv(config.feature_root / "train_features_raw.csv")
    test = pd.read_csv(config.feature_root / "test_features_raw.csv")
    leakage_report = audit_leakage(train, test)
    filter_result = fit_training_feature_filter(train[FEATURE_COLUMNS], FEATURE_COLUMNS, config)
    top_k_result = evaluate_top_k_grouped(train, filter_result.kept_features, config)
    metrics = train_and_evaluate_models(train, test, top_k_result.selected_features, config)
    summary = {
        "leakage_check": leakage_report,
        "selected_k": top_k_result.selected_k,
        "selected_features": top_k_result.selected_features,
        "removed_features": filter_result.removed_reasons,
        "metrics_path": str(config.output_root / "model_metrics.csv"),
    }
    (config.output_root / "run_complete.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
```

- [ ] **Step 4: Run the grouped modeling tests**

Run: `pytest tests/test_grouped_modeling.py -v`

Expected: PASS

- [ ] **Step 5: Commit the modeling stage**

```bash
git add pump_diagnosis/modeling.py pump_diagnosis/model_training.py pump_diagnosis/modeling_runner.py tests/test_grouped_modeling.py
git commit -m "feat: add grouped model selection and final training"
```

## Task 6: Wire the CLI, Compatibility Entrypoints, README, and Dependency Checks

**Files:**
- Create: `pump_diagnosis/runner.py`
- Modify: `train_models.py`
- Create: `README.md`
- Modify: `tests/test_rebuild_outputs.py`

- [ ] **Step 1: Write the failing CLI and lazy-import tests**

```python
def test_importing_runner_does_not_import_matplotlib():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import pump_diagnosis.runner; print('matplotlib' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"
```

```python
def test_runner_cli_supports_stage_names(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "pump_diagnosis.runner", "--stage", "inspect", "--output-root", str(tmp_path / "output")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
```

- [ ] **Step 2: Run the CLI tests**

Run: `pytest tests/test_rebuild_outputs.py::test_importing_runner_does_not_import_matplotlib tests/test_rebuild_outputs.py::test_runner_cli_supports_stage_names -v`

Expected: FAIL until the runner defers plot imports and exposes the documented stage names.

- [ ] **Step 3: Implement the CLI contract and documentation**

```python
# pump_diagnosis/runner.py
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inspect", "split", "preprocess", "features", "select", "train", "evaluate", "all"], required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    config = PipelineConfig(output_root=Path(args.output_root)) if args.output_root else PipelineConfig()
    if args.stage == "inspect":
        run_inspect_stage(config)
    elif args.stage == "split":
        run_split_stage(config)
    elif args.stage == "all":
        run_all_stages(config)
    else:
        run_single_stage(args.stage, config)


if __name__ == "__main__":
    main()
```

```python
# train_models.py
from pump_diagnosis.runner import main


if __name__ == "__main__":
    main()
```

```markdown
# README.md
## 运行顺序
1. `python -m pump_diagnosis.runner --stage inspect`
2. `python -m pump_diagnosis.runner --stage split`
3. `python -m pump_diagnosis.runner --stage all`

## 特征总览
- 18 个时域特征
- 20 个基础频域特征
- 9 个转频特征
- 17 个三级 db4 小波包特征
- 20 个包络与包络谱特征
```

- [ ] **Step 4: Run the output-contract tests**

Run: `pytest tests/test_rebuild_outputs.py -v`

Expected: PASS

- [ ] **Step 5: Commit the interface layer**

```bash
git add pump_diagnosis/runner.py train_models.py README.md tests/test_rebuild_outputs.py
git commit -m "feat: expose runnable pipeline cli"
```

## Task 7: Run Full Verification and Real-Data Audit

**Files:**
- Modify: `README.md`
- Modify: `outputs/channel4_six_class_v1/` (generated artifacts only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`

Expected: PASS for all repository tests.

- [ ] **Step 2: Run the real data inspection and split stages**

Run: `python -m pump_diagnosis.runner --stage inspect`

Expected: PASS and write `data_inventory.csv`, `data_quality_report.csv`, `label_mapping.json`, and `training_pipeline.log` under `outputs/channel4_six_class_v1/`.

Run: `python -m pump_diagnosis.runner --stage split`

Expected: PASS and write train/test manifests plus zero-overlap leakage audit JSON.

- [ ] **Step 3: Run the full pipeline once on the real dataset**

Run: `python -m pump_diagnosis.runner --stage all`

Expected: PASS and write raw features, selected features, scaler, three model files, confusion matrices, reports, and run-complete markers.

- [ ] **Step 4: Verify output completeness against the spec**

Run: `python -m pump_diagnosis.runner --stage evaluate`

Expected: PASS and produce final metrics without retraining or changing the test predictions.

Run: `rg -n "TODO|TBD|pass #|NotImplemented" pump_diagnosis tests README.md`

Expected: no matches

- [ ] **Step 5: Commit the verified implementation**

```bash
git add README.md pump_diagnosis tests config.py train_models.py requirements.txt outputs/channel4_six_class_v1
git commit -m "feat: deliver channel4 six-class diagnosis pipeline"
```

## Self-Review

### Spec Coverage

- Channel-4-only discovery is covered in Task 2.
- Source-file-level 80/20 split, duplicate-hash grouping, and leakage audit are covered in Task 2 and Task 5.
- DC removal, 20 kHz to 12 kHz resampling, 10-5000 Hz filtering, and 4096/2048 windows are covered in Task 3.
- Full multi-domain feature extraction and feature dictionary generation are covered in Task 3 and Task 4.
- Training-only invalid-feature removal, correlation filtering, standardization, grouped CV, and three-model comparison are covered in Task 5.
- CLI stages, output contracts, README, and reproducibility are covered in Task 6.
- Full test-suite run and one real-data execution with audit artifacts are covered in Task 7.

### Placeholder Scan

- Searched manually for `TODO`, `TBD`, `implement later`, `similar to Task`, and unresolved ellipsis in task instructions.
- No task says “write tests” or “handle errors” without concrete files, commands, and expected outcomes.

### Type Consistency

- `PipelineConfig`, `ModelingConfig`, `ConditionCatalog`, `FEATURE_COLUMNS`, `METADATA_COLUMNS`, `extract_split_features`, `validate_feature_csv`, `fit_training_feature_filter`, `evaluate_top_k_grouped`, `train_and_evaluate_models`, and `run_full_modeling` use the same names throughout the plan.
- The split artifact names `train_features_raw.csv` and `test_features_raw.csv` are used consistently between Task 4 and Task 5.
- The CLI stage names are fixed once in Task 6 and reused in Task 7.
