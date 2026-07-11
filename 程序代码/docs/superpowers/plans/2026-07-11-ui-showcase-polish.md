# UI Showcase Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Streamlit presentation for thesis demonstration without changing diagnosis behavior or data contracts.

**Architecture:** Keep all inference and service calls unchanged. Move display-only transformations into `pump_fault_app.presentation`, while page modules retain layout, input collection, and rendering responsibilities.

**Tech Stack:** Python, Streamlit, Altair, pandas, pytest.

---

### Task 1: Define display-only view helpers

**Files:**
- Create: `pump_fault_app/presentation/batch_diagnosis.py`
- Create: `pump_fault_app/presentation/report_view.py`
- Modify: `pump_fault_app/presentation/single_diagnosis.py`
- Test: `tests/test_pump_fault_app_ui.py`

- [ ] Write failing tests for upload information, batch task statistics, and diagnosis highlight states.
- [ ] Run the focused UI tests and confirm the helpers are missing.
- [ ] Add minimal pure helpers that only format existing result data.
- [ ] Re-run focused UI tests.

### Task 2: Apply page-level visual hierarchy

**Files:**
- Modify: `pump_fault_app/ui/streamlit_app.py`
- Modify: `pump_fault_app/ui/pages/single_diagnosis.py`
- Modify: `pump_fault_app/ui/pages/batch_diagnosis.py`
- Modify: `pump_fault_app/ui/pages/report_view.py`
- Test: `tests/test_pump_fault_app_ui.py`

- [ ] Write failing UI-source assertions for the new menu label and presentation-helper usage.
- [ ] Implement homepage capability cards and vertical research flow, upload information, batch metrics, and result status cards.
- [ ] Run focused UI tests.

### Task 3: Document and verify

**Files:**
- Modify: `docs/pump_fault_app_architecture.md`
- Test: `tests/test_pump_fault_app_ui.py`

- [ ] Update the architecture document with UI/presentation boundaries.
- [ ] Run `../.venv/bin/python -m pytest tests/ -q`.
- [ ] Start the Streamlit app and validate that the home page is reachable.
