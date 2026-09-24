from __future__ import annotations

from pathlib import Path


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY_PATH = PROGRAM_ROOT / "runtime_data" / "pump_fault_model_registry.sqlite3"
DEFAULT_CANDIDATE_MODEL_DIR = PROGRAM_ROOT / "model_training"
