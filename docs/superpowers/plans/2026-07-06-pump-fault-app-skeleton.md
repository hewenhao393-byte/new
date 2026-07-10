# Pump Fault App Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated software skeleton package for six-class pump fault diagnosis with unified configuration, domain data structures, logging, and future-facing interfaces.

**Architecture:** Create a new `pump_fault_app` package under `程序代码/` and keep it separate from the existing `pump_diagnosis` experiment code. Centralize all fixed parameters in config dataclasses, expose stable domain records and protocols, and provide a bootstrap entrypoint that proves configuration and logging work without implementing training or UI.

**Tech Stack:** Python 3, dataclasses, pathlib, logging, typing.Protocol, pytest

---

### Task 1: Add contract tests for configuration and bootstrap

**Files:**
- Create: `程序代码/tests/test_pump_fault_app_config.py`
- Create: `程序代码/tests/test_pump_fault_app_logging.py`
- Test: `程序代码/tests/test_pump_fault_app_config.py`
- Test: `程序代码/tests/test_pump_fault_app_logging.py`

- [ ] Write failing tests for fixed parameters, label order, feature order, bootstrap summary, and log file output.
- [ ] Run the targeted tests and confirm they fail because `pump_fault_app` does not exist yet.

### Task 2: Implement config and domain modules

**Files:**
- Create: `程序代码/pump_fault_app/__init__.py`
- Create: `程序代码/pump_fault_app/config/defaults.py`
- Create: `程序代码/pump_fault_app/config/schema.py`
- Create: `程序代码/pump_fault_app/config/loader.py`
- Create: `程序代码/pump_fault_app/domain/labels.py`
- Create: `程序代码/pump_fault_app/domain/features.py`
- Create: `程序代码/pump_fault_app/domain/records.py`

- [ ] Add minimal code to satisfy the configuration and data-contract tests.
- [ ] Re-run the configuration tests and confirm they pass.

### Task 3: Implement logging and future interfaces

**Files:**
- Create: `程序代码/pump_fault_app/logging/context.py`
- Create: `程序代码/pump_fault_app/logging/setup.py`
- Create: `程序代码/pump_fault_app/interfaces/feature_extractor.py`
- Create: `程序代码/pump_fault_app/interfaces/classifier.py`
- Create: `程序代码/pump_fault_app/interfaces/repository.py`
- Create: `程序代码/pump_fault_app/adapters/legacy/contracts.py`
- Create: `程序代码/pump_fault_app/app/bootstrap.py`

- [ ] Implement minimal logging setup, context adapter, protocol interfaces, and bootstrap entrypoint.
- [ ] Run the logging/bootstrap tests and confirm they pass.

### Task 4: Document how to run and test

**Files:**
- Modify: `程序代码/README.md`

- [ ] Add a short section describing how to run the bootstrap command and how to execute the new tests.
- [ ] Run the targeted test suite one more time and verify all new tests are green.
