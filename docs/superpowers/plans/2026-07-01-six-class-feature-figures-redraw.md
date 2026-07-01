# Six-Class Feature Figures Redraw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Chapter 4 six-class feature figures from window-level representatives, export new figure assets and metadata, and replace the corresponding figures and prose in the desktop thesis DOCX with render-verified layout.

**Architecture:** Add a new figure-generation module alongside the existing V2 paper figure script so the redraw pipeline can reuse stable preprocessing helpers while replacing the old device-split layout completely. Then update the desktop DOCX in a new output file by replacing only the Chapter 4 figure blocks, renumbering downstream figures, and validating the result through full-page rendering.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, matplotlib, seaborn, python-docx, pypdf, pytest, LibreOffice headless renderer

---

### Task 1: Add tests for representative-window selection and redraw exports

**Files:**
- Create: `程序代码/tests/test_six_class_feature_redraw.py`
- Modify: `程序代码/pump_diagnosis/six_class_feature_redraw.py`
- Test: `程序代码/tests/test_six_class_feature_redraw.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import numpy as np
import pandas as pd

from pump_diagnosis.six_class_feature_redraw import (
    CLASS_LAYOUT_ORDER,
    REDRAW_FEATURE_COLUMNS,
    normalize_window_waveform,
    select_representative_windows,
    should_expand_envelope_limit,
)


def test_select_representative_windows_returns_one_window_per_class():
    frame = pd.DataFrame(
        [
            {"label": "正常", "device_id": "Motor-2", "speed_percent": 100, "rpm": 1480, "source_file": "a.csv", "group_id": "g1", "window_id": "w1", **{name: 0.0 for name in REDRAW_FEATURE_COLUMNS}},
            {"label": "正常", "device_id": "Motor-2", "speed_percent": 100, "rpm": 1480, "source_file": "a.csv", "group_id": "g1", "window_id": "w2", **{name: 1.0 for name in REDRAW_FEATURE_COLUMNS}},
            {"label": "转子不平衡", "device_id": "Motor-4", "speed_percent": 70, "rpm": 2070, "source_file": "b.csv", "group_id": "g2", "window_id": "w3", **{name: 2.0 for name in REDRAW_FEATURE_COLUMNS}},
            {"label": "转子不平衡", "device_id": "Motor-4", "speed_percent": 70, "rpm": 2070, "source_file": "b.csv", "group_id": "g2", "window_id": "w4", **{name: 3.0 for name in REDRAW_FEATURE_COLUMNS}},
        ]
    )
    selected = select_representative_windows(frame)
    assert set(selected) == {"正常", "转子不平衡"}
    assert selected["正常"]["window_id"] == "w1"
    assert selected["转子不平衡"]["window_id"] == "w3"


def test_normalize_window_waveform_scales_to_unit_range():
    waveform = np.array([0.0, 2.0, -4.0, 1.0], dtype=float)
    normalized = normalize_window_waveform(waveform)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)
    assert np.isclose(normalized[2], -1.0)


def test_should_expand_envelope_limit_only_when_peak_exceeds_300():
    assert not should_expand_envelope_limit([120.0, 180.0, 250.0])
    assert should_expand_envelope_limit([120.0, 180.0, 305.0])


def test_layout_order_matches_required_2x3_sequence():
    assert CLASS_LAYOUT_ORDER == [
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd '/Users/hewenhao/Documents/特征提取/程序代码' && ../.venv/bin/python -m pytest tests/test_six_class_feature_redraw.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `pump_diagnosis.six_class_feature_redraw`

- [ ] **Step 3: Write minimal implementation**

```python
# 程序代码/pump_diagnosis/six_class_feature_redraw.py
CLASS_LAYOUT_ORDER = [
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
]

REDRAW_FEATURE_COLUMNS = [
    "kurtosis",
    "skewness",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "rot_2x_1x_ratio",
    "rot_3x_1x_ratio",
    "harmonic_energy_ratio_1x_5x",
    "spectral_entropy",
    "spectral_flatness",
    "wp_energy_ratio_0",
    "wp_energy_ratio_1",
    "wp_energy_ratio_2",
    "wp_energy_ratio_3",
    "wp_energy_ratio_4",
    "wp_energy_ratio_5",
    "wp_energy_ratio_6",
    "wp_energy_ratio_7",
    "env_kurtosis",
    "env_crest_factor",
]


def select_representative_windows(frame):
    ...


def normalize_window_waveform(waveform):
    ...


def should_expand_envelope_limit(peaks_hz):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd '/Users/hewenhao/Documents/特征提取/程序代码' && ../.venv/bin/python -m pytest tests/test_six_class_feature_redraw.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd '/Users/hewenhao/Documents/特征提取'
git add 程序代码/tests/test_six_class_feature_redraw.py 程序代码/pump_diagnosis/six_class_feature_redraw.py
git commit -m "Add tests for six-class redraw selection helpers"
```

### Task 2: Implement the redraw pipeline for five unified 2x3 figures

**Files:**
- Create: `程序代码/pump_diagnosis/six_class_feature_redraw.py`
- Modify: `程序代码/pump_diagnosis/fault_feature_visualization.py`
- Test: `程序代码/tests/test_six_class_feature_redraw.py`

- [ ] **Step 1: Extend the test with export and metadata expectations**

```python
from pathlib import Path

from pump_diagnosis.six_class_feature_redraw import RedrawConfig, write_representative_window_manifest


def test_write_representative_window_manifest_preserves_required_columns(tmp_path: Path):
    rows = [
        {
            "label": "正常",
            "device_id": "Motor-2",
            "speed_percent": 100,
            "rpm": 1480.0,
            "source_file": "x.csv",
            "group_id": "g1",
            "window_id": "w1",
        }
    ]
    output = tmp_path / "representative_windows.csv"
    write_representative_window_manifest(rows, output)
    frame = pd.read_csv(output)
    assert frame.columns.tolist()[:7] == [
        "label",
        "device_id",
        "speed_percent",
        "rpm",
        "source_file",
        "group_id",
        "window_id",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd '/Users/hewenhao/Documents/特征提取/程序代码' && ../.venv/bin/python -m pytest tests/test_six_class_feature_redraw.py -q`
Expected: FAIL with missing `RedrawConfig` or `write_representative_window_manifest`

- [ ] **Step 3: Implement the redraw pipeline**

```python
@dataclass(frozen=True)
class RedrawConfig:
    output_root: Path
    processed_fs: int = 12_000
    window_size: int = 2400
    window_step: int = 1200
    fft_low_max_hz: float = 300.0
    fft_full_max_hz: float = 5000.0
    envelope_default_max_hz: float = 300.0
    envelope_expanded_max_hz: float = 500.0
    wavelet: str = "db6"
    wavelet_level: int = 3


def run_six_class_feature_redraw(config: RedrawConfig) -> dict[str, str]:
    """Load unified six-class source tables, select representative windows, rebuild
    exact 2400-point processed windows, export CSVs, and save five 2x3 figure pages."""
```

Implementation requirements for this task:

- Load the four existing source feature tables used by the unified six-class experiment.
- Reconstruct global `device_id`, `speed_percent`, `rpm`, `group_id`, and `window_id`.
- Select one representative window per class using the 21-feature center distance.
- Rebuild the exact processed 2400-point window from the raw source file using `window_id`.
- Export:
  - `representative_windows.csv`
  - one CSV per figure
  - five figure PNG/PDF pairs in a new result directory under `实验结果/`
- Plot five figure pages:
  - `fig4_2_time_waveform`
  - `fig4_3_low_frequency_spectrum`
  - `fig4_4_full_spectrum`
  - `fig4_5_wavelet_packet_bar`
  - `fig4_6_envelope_spectrum`

- [ ] **Step 4: Run tests and the redraw script**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python -m pytest tests/test_six_class_feature_redraw.py -q
../.venv/bin/python -m pump_diagnosis.six_class_feature_redraw
```

Expected:

- pytest PASS
- script prints output paths for 5 figures and `representative_windows.csv`

- [ ] **Step 5: Commit**

```bash
cd '/Users/hewenhao/Documents/特征提取'
git add 程序代码/pump_diagnosis/six_class_feature_redraw.py 程序代码/tests/test_six_class_feature_redraw.py
git commit -m "Implement unified six-class feature redraw pipeline"
```

### Task 3: Verify figure outputs and inspect layout data before touching the DOCX

**Files:**
- Modify: `程序代码/pump_diagnosis/six_class_feature_redraw.py`
- Test: `实验结果/six_class_feature_redraw/`

- [ ] **Step 1: Add a validation helper for figure output completeness**

```python
def validate_redraw_outputs(output_root: Path) -> None:
    required = [
        "representative_windows.csv",
        "fig4_2_time_waveform.png",
        "fig4_2_time_waveform.pdf",
        "fig4_3_low_frequency_spectrum.png",
        "fig4_4_full_spectrum.png",
        "fig4_5_wavelet_packet_bar.png",
        "fig4_6_envelope_spectrum.png",
    ]
    missing = [name for name in required if not (output_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing redraw outputs: {missing}")
```

- [ ] **Step 2: Run the redraw script and validate the output directory**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python -m pump_diagnosis.six_class_feature_redraw
../.venv/bin/python - <<'PY'
from pathlib import Path
from pump_diagnosis.six_class_feature_redraw import validate_redraw_outputs
validate_redraw_outputs(Path('/Users/hewenhao/Documents/特征提取/实验结果/six_class_feature_redraw'))
print('OK')
PY
```

Expected: `OK`

- [ ] **Step 3: Inspect representative window metadata and envelope limit decision**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python - <<'PY'
import pandas as pd
root = '/Users/hewenhao/Documents/特征提取/实验结果/six_class_feature_redraw'
frame = pd.read_csv(f'{root}/representative_windows.csv')
print(frame[['label','device_id','speed_percent','rpm','group_id','window_id']].to_string(index=False))
PY
```

Expected: exactly six rows, one for each class, with non-empty metadata

- [ ] **Step 4: Commit**

```bash
cd '/Users/hewenhao/Documents/特征提取'
git add 程序代码/pump_diagnosis/six_class_feature_redraw.py
git commit -m "Add redraw output validation helpers"
```

### Task 4: Replace Chapter 4 feature figures and renumber downstream figures in the desktop DOCX

**Files:**
- Modify: `/Users/hewenhao/Desktop/水泵故障诊断论文_V2六分类图件补全版.docx`
- Create: `实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx`
- Test: `实验结果/docx_render_六类特征图重绘替换版/`

- [ ] **Step 1: Write a deterministic DOCX replacement script**

```python
from docx import Document


def replace_chapter4_feature_figures(src_docx: str, dst_docx: str, figure_root: str) -> None:
    """Replace the old Chapter 4 feature figures, update captions 4-2 to 4-6,
    renumber downstream figures, and revise the nearby prose blocks."""
```

The script must:

- read the desktop DOCX as source
- write a new DOCX under `实验结果/`
- replace only the old Chapter 4 feature figure blocks
- update captions to new `图4-2` through `图4-6`
- renumber old model figures/captions to start from `图4-7`
- replace old phrases:
  - `按两个设备分别展示`
  - `低频和全频频谱放在同一张图`
  - `小波包热力图`
- inject the approved new spectrum and wavelet paragraphs
- keep the rest of the document intact

- [ ] **Step 2: Run the replacement script and produce a new DOCX**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python -m pump_diagnosis.six_class_feature_redraw --update-docx \
  --input-docx '/Users/hewenhao/Desktop/水泵故障诊断论文_V2六分类图件补全版.docx' \
  --output-docx '/Users/hewenhao/Documents/特征提取/实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx'
```

Expected: new DOCX file created successfully

- [ ] **Step 3: Render the new DOCX for QA**

Run:

```bash
env TMPDIR=/private/tmp \
'/Users/hewenhao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' \
'/Users/hewenhao/.codex/plugins/cache/openai-primary-runtime/documents/26.623.12021/skills/documents/render_docx.py' \
'/Users/hewenhao/Documents/特征提取/实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx' \
--output_dir '/Users/hewenhao/Documents/特征提取/实验结果/docx_render_六类特征图重绘替换版' \
--emit_pdf
```

Expected: page PNGs and a PDF are created

- [ ] **Step 4: Commit**

```bash
cd '/Users/hewenhao/Documents/特征提取'
git add 程序代码/pump_diagnosis/six_class_feature_redraw.py
git commit -m "Replace chapter 4 feature figures in thesis docx"
```

### Task 5: Perform full visual QA of Chapter 4 and fix layout defects

**Files:**
- Modify: `程序代码/pump_diagnosis/six_class_feature_redraw.py`
- Modify: `实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx`
- Test: `实验结果/docx_render_六类特征图重绘替换版/`

- [ ] **Step 1: Inspect all rendered Chapter 4 pages**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取'
ls 实验结果/docx_render_六类特征图重绘替换版/page-*.png
```

Expected: identify the Chapter 4 page range and inspect each page for:

- one large feature figure per page where possible
- no old device-split figure remnants
- correct figure numbers `图4-2` to `图4-9` or later
- no clipped axes, captions, or overlapping paragraphs

- [ ] **Step 2: Fix layout or wording issues and re-render**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python -m pump_diagnosis.six_class_feature_redraw --update-docx \
  --input-docx '/Users/hewenhao/Desktop/水泵故障诊断论文_V2六分类图件补全版.docx' \
  --output-docx '/Users/hewenhao/Documents/特征提取/实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx'
env TMPDIR=/private/tmp \
'/Users/hewenhao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' \
'/Users/hewenhao/.codex/plugins/cache/openai-primary-runtime/documents/26.623.12021/skills/documents/render_docx.py' \
'/Users/hewenhao/Documents/特征提取/实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx' \
--output_dir '/Users/hewenhao/Documents/特征提取/实验结果/docx_render_六类特征图重绘替换版' \
--emit_pdf
```

Expected: corrected render output with no visible Chapter 4 defects

- [ ] **Step 3: Final verification commands**

Run:

```bash
cd '/Users/hewenhao/Documents/特征提取/程序代码'
../.venv/bin/python -m pytest tests/test_six_class_feature_redraw.py tests/test_v2_paper_figures.py -q
../.venv/bin/python - <<'PY'
from docx import Document
doc = Document('/Users/hewenhao/Documents/特征提取/实验结果/水泵故障诊断论文_六类特征图重绘替换版.docx')
hits = []
for p in doc.paragraphs:
    text = ''.join(run.text for run in p.runs)
    if any(flag in text for flag in ['按两个设备分别展示', '低频和全频频谱放在同一张图', '小波包热力图', '【此处插入']):
        hits.append(text)
print(hits)
PY
```

Expected:

- pytest PASS
- `[]` from the DOCX text scan

- [ ] **Step 4: Commit**

```bash
cd '/Users/hewenhao/Documents/特征提取'
git add 程序代码/pump_diagnosis/six_class_feature_redraw.py 程序代码/tests/test_six_class_feature_redraw.py
git commit -m "Finalize six-class figure redraw and docx QA"
```

## Self-Review

- Spec coverage:
  - Representative window selection: Task 1 and Task 2
  - Five new unified figures with exact layout and export formats: Task 2 and Task 3
  - DOCX replacement, renumbering, and prose cleanup: Task 4
  - Render-based visual QA and residual-text scan: Task 5
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to above” markers remain.
- Type consistency:
  - The plan consistently uses `select_representative_windows`, `normalize_window_waveform`, `should_expand_envelope_limit`, `RedrawConfig`, `run_six_class_feature_redraw`, and `replace_chapter4_feature_figures`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-six-class-feature-figures-redraw.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
