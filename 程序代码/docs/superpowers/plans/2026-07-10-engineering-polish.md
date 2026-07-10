# Pump Fault App Engineering Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve software ownership boundaries and presentation quality without changing the formal BP inference behavior.

**Architecture:** Move the frozen V2 contract behind an application-owned compatibility module while preserving the original experiment import path. Centralize software-facing versions, expose them in self-check and Word reports, and extract UI display adapters that only format existing service results.

**Tech Stack:** Python 3, dataclasses, pytest, Streamlit, python-docx.

---

### Task 1: Formal Contract and Version Ownership

**Files:**
- Create: `pump_fault_app/domain/formal_contract.py`
- Create: `pump_fault_app/version.py`
- Modify: `pump_diagnosis/inference_contract.py`
- Modify: application modules currently importing `pump_diagnosis.inference_contract`
- Test: `tests/test_pump_fault_app_config.py`

- [ ] Add failing tests that assert application-owned contract imports expose the unchanged V2 feature names, labels and signal parameters, and that the centralized version constants are non-empty.
- [ ] Run the focused tests and confirm they fail because the new modules do not exist.
- [ ] Add the compatibility contract module and centralized version module; redirect application imports to the application-owned contract while retaining legacy exports for experiment scripts.
- [ ] Run focused tests and confirm they pass.

### Task 2: Versioned Reporting and Self-Check

**Files:**
- Modify: `pump_fault_app/app/self_check.py`
- Modify: `pump_fault_app/reporting/view_data.py`
- Modify: `pump_fault_app/reporting/word_export.py`
- Test: `tests/test_pump_fault_app_self_check.py`
- Test: `tests/test_pump_fault_app_reporting.py`

- [ ] Add failing tests requiring a structured `version_information` self-check item and report view data / Word metadata sourced from `pump_fault_app.version`.
- [ ] Run focused tests and confirm expected failures.
- [ ] Replace direct report and self-check version reads with the centralized version module.
- [ ] Run focused tests and confirm they pass.

### Task 3: Presentation Adapters and Alert Copy

**Files:**
- Create: `pump_fault_app/presentation/single_diagnosis.py`
- Create: `pump_fault_app/presentation/__init__.py`
- Modify: `pump_fault_app/ui/pages/single_diagnosis.py`
- Modify: `pump_fault_app/ui/pages/report_view.py`
- Modify: `pump_fault_app/reporting/view_data.py`
- Test: `tests/test_pump_fault_app_ui.py`
- Test: `tests/test_pump_fault_app_reporting.py`

- [ ] Add failing tests requiring a neutral processing-status message for runtime alerts while preserving the exact alert count in detailed report data, and requiring UI helper imports to use the presentation layer.
- [ ] Run focused tests and confirm expected failures.
- [ ] Move row construction and alert presentation formatting into the presentation module, keeping UI pages responsible for Streamlit layout and drawing only.
- [ ] Run focused tests and confirm they pass.

### Task 4: Architecture Documentation and Regression Verification

**Files:**
- Create: `docs/pump_fault_app_architecture.md`
- Modify: `README.md`
- Test: `tests/test_pump_fault_app_self_check.py`

- [ ] Add a failing document-presence and content test for the architecture documentation and formal V2 parameters.
- [ ] Run the focused test and confirm expected failure.
- [ ] Document layers, data flow, contract ownership, UI boundaries and the unchanged formal inference constraints; link it from the README.
- [ ] Run the requested full pytest suite and confirm no regression in inference, UI, reporting or export behavior.
