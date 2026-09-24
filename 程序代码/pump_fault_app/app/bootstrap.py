from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pump_fault_app.config.loader import build_default_config
from pump_fault_app.config.schema import AppConfig
from pump_fault_app.domain.formal_contract import FORMAL_FEATURE_NAMES, FORMAL_LABEL_ORDER, FORMAL_V3_CONTRACT
from pump_fault_app.logging.context import LoggingContext
from pump_fault_app.logging.setup import setup_logging


@dataclass(frozen=True)
class BootstrapState:
    config: AppConfig
    context: LoggingContext
    summary: dict[str, object]


def bootstrap_application(project_root: Path | None = None) -> BootstrapState:
    config = build_default_config(project_root=project_root)
    context = LoggingContext(run_id=f"run-{uuid4().hex[:8]}", component="bootstrap")
    logger, log_file = setup_logging(config.logging, config.paths.runtime_root, context)
    summary = {
        "label_count": len(FORMAL_LABEL_ORDER),
        "feature_count": len(FORMAL_FEATURE_NAMES),
        "target_sample_rate_hz": FORMAL_V3_CONTRACT.target_sampling_rate,
        "window_size": FORMAL_V3_CONTRACT.window_size,
        "step_size": FORMAL_V3_CONTRACT.step_size,
        "log_directory": str(config.paths.runtime_root),
        "log_file": str(log_file),
    }
    logger.info("pump_fault_app bootstrap complete")
    return BootstrapState(config=config, context=context, summary=summary)


def main() -> None:
    state = bootstrap_application()
    print(json.dumps(state.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
