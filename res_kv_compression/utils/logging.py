"""Logging helpers for reproducible experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from utils.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure process logging and return the project logger."""

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger("res_kv_compression")
    logger.setLevel(log_level)
    return logger


class ExperimentLogger:
    """File-backed JSON logger for configs and scalar metrics."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.config_path = self.output_dir / "config.json"

    def log_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        payload: dict[str, Any] = dict(metrics)
        if step is not None:
            payload["step"] = step
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
