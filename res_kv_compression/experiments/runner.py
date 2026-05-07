"""Experiment runner boundary.

The runner is intentionally thin in Phase 1. Later phases register concrete
compression and evaluation components behind this stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.config import ExperimentConfig, config_to_dict
from utils.logging import ExperimentLogger, setup_logging
from utils.seed import set_random_seed


@dataclass(frozen=True)
class ExperimentResult:
    """Serializable result returned by experiment entrypoints."""

    experiment: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a configured experiment.

    Model execution is deferred until at least one model-backed evaluation flag
    is enabled. This keeps setup and unit tests reproducible on CPU-only
    machines and offline workstations.
    """

    set_random_seed(config.experiment.seed)
    logger = setup_logging(config.logging)
    experiment_logger = ExperimentLogger(config.logging.output_dir)

    logger.info("Starting experiment %s", config.experiment.name)
    experiment_logger.log_config(config_to_dict(config))

    metadata: dict[str, Any] = {
        "model_name": config.model.model_name,
        "compression_rank": config.compression.rank,
        "quant_bits": config.quantization.quant_bits,
    }

    model_backed_eval = any(
        (
            config.evaluation.perplexity_eval,
            config.evaluation.long_context_eval,
            config.evaluation.latency_eval,
            config.evaluation.memory_eval,
        )
    )
    if model_backed_eval:
        metadata["status"] = "model_evaluation_not_registered_yet"
        logger.info("Model-backed evaluators will be registered in later phases")
    else:
        metadata["status"] = "setup_complete"

    result = ExperimentResult(
        experiment=config.experiment.name,
        metrics={},
        artifacts={"output_dir": str(config.logging.output_dir)},
        metadata=metadata,
    )
    experiment_logger.log_metrics({"setup_complete": 1.0})
    return result
