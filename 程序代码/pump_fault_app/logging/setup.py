from __future__ import annotations

import logging
from pathlib import Path

from pump_fault_app.config.schema import LoggingConfig
from pump_fault_app.logging.context import LoggingContext


def setup_logging(config: LoggingConfig, runtime_root: Path, context: LoggingContext) -> tuple[logging.LoggerAdapter, Path]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_file = runtime_root / config.log_filename
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(run_id)s | %(component)s | %(label_space)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    adapter = logging.LoggerAdapter(
        logger,
        {
            "run_id": context.run_id,
            "component": context.component,
            "label_space": context.label_space,
        },
    )
    return adapter, log_file
