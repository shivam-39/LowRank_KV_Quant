"""Typed YAML configuration for experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentInfoConfig:
    name: str = "low_rank_residual_kv"
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"
    device: str = "auto"
    max_seq_len: int = 2048
    trust_remote_code: bool = False
    use_cache: bool = True


@dataclass(frozen=True)
class CompressionConfig:
    rank: int = 16
    decomposition_type: str = "truncated_svd"
    randomized_oversamples: int = 8
    randomized_n_iter: int = 2
    adaptive_rank: bool = False
    shared_rank: bool = True
    energy_threshold: float = 0.99
    granularity: str = "per_head"


@dataclass(frozen=True)
class QuantizationConfig:
    quant_bits: int = 4
    group_size: int = 64
    per_channel: bool = False
    symmetric: bool = True
    fake_quant: bool = True


@dataclass(frozen=True)
class AttentionConfig:
    mode: str = "reconstructed"
    causal: bool = True
    dropout_p: float = 0.0


@dataclass(frozen=True)
class ObjectiveConfig:
    use_attention_loss: bool = True
    use_softmax_loss: bool = True
    use_output_loss: bool = True
    softmax_loss_type: str = "kl"
    lambda_recon: float = 1.0
    lambda_attention: float = 1.0
    lambda_softmax: float = 1.0
    lambda_output: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    perplexity_eval: bool = False
    long_context_eval: bool = False
    latency_eval: bool = False
    memory_eval: bool = False
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    max_eval_samples: int = 32
    batch_size: int = 1
    stride: int = 512
    prompt: str = "Low-rank residual KV cache compression"
    latency_warmup: int = 1
    latency_iters: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    wandb: bool = False
    tensorboard: bool = False
    output_dir: str = "logs/default"
    log_level: str = "INFO"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentInfoConfig = ExperimentInfoConfig()
    model: ModelConfig = ModelConfig()
    compression: CompressionConfig = CompressionConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    attention: AttentionConfig = AttentionConfig()
    objectives: ObjectiveConfig = ObjectiveConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()


_SECTION_TYPES = {
    "experiment": ExperimentInfoConfig,
    "model": ModelConfig,
    "compression": CompressionConfig,
    "quantization": QuantizationConfig,
    "attention": AttentionConfig,
    "objectives": ObjectiveConfig,
    "evaluation": EvaluationConfig,
    "logging": LoggingConfig,
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML into typed dataclasses."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping")
    return _build_config(payload)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert typed config into a plain dictionary."""

    return asdict(config)


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply Hydra-style dotted overrides such as ``compression.rank=8``."""

    payload = config_to_dict(config)
    for override in overrides:
        key, value = _split_override(override)
        _set_nested(payload, key.split("."), _parse_scalar(value))
    return _build_config(payload)


def _build_config(payload: dict[str, Any]) -> ExperimentConfig:
    sections: dict[str, Any] = {}
    unknown_sections = set(payload) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    for section_name, section_type in _SECTION_TYPES.items():
        section_payload = payload.get(section_name, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        defaults = asdict(section_type())
        unknown_keys = set(section_payload) - set(defaults)
        if unknown_keys:
            raise ValueError(f"Unknown keys in section '{section_name}': {sorted(unknown_keys)}")
        sections[section_name] = section_type(**{**defaults, **section_payload})

    config = ExperimentConfig(**sections)
    _validate_config(config)
    return config


def _split_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"Override '{override}' must use key=value syntax")
    key, value = override.split("=", 1)
    if not key:
        raise ValueError(f"Override '{override}' has an empty key")
    return key, value


def _set_nested(payload: dict[str, Any], path: list[str], value: Any) -> None:
    cursor: dict[str, Any] = payload
    for part in path[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ValueError(f"Unknown override path: {'.'.join(path)}")
        cursor = cursor[part]
    leaf = path[-1]
    if leaf not in cursor:
        raise ValueError(f"Unknown override key: {'.'.join(path)}")
    cursor[leaf] = value


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _validate_config(config: ExperimentConfig) -> None:
    if config.compression.rank <= 0:
        raise ValueError("compression.rank must be positive")
    if not 0.0 < config.compression.energy_threshold <= 1.0:
        raise ValueError("compression.energy_threshold must be in (0, 1]")
    if config.compression.decomposition_type not in {"truncated_svd", "randomized_svd", "pca"}:
        raise ValueError("compression.decomposition_type must be truncated_svd, randomized_svd, or pca")
    if config.compression.granularity not in {"shared", "per_layer", "per_head"}:
        raise ValueError("compression.granularity must be shared, per_layer, or per_head")
    if config.quantization.quant_bits not in {2, 4, 8}:
        raise ValueError("quantization.quant_bits must be one of {2, 4, 8}")
    if config.quantization.group_size <= 0:
        raise ValueError("quantization.group_size must be positive")
    if config.attention.mode not in {"reconstructed", "low_rank_only", "hybrid", "baseline"}:
        raise ValueError("attention.mode must be reconstructed, low_rank_only, hybrid, or baseline")
    if config.objectives.softmax_loss_type not in {"kl", "mse"}:
        raise ValueError("objectives.softmax_loss_type must be kl or mse")
    if config.evaluation.batch_size <= 0:
        raise ValueError("evaluation.batch_size must be positive")
    if config.evaluation.stride <= 0:
        raise ValueError("evaluation.stride must be positive")
    if config.evaluation.latency_warmup < 0:
        raise ValueError("evaluation.latency_warmup must be non-negative")
    if config.evaluation.latency_iters <= 0:
        raise ValueError("evaluation.latency_iters must be positive")


def with_logging_dir(config: ExperimentConfig, output_dir: str) -> ExperimentConfig:
    """Return a config copy with a different logging directory."""

    return replace(config, logging=replace(config.logging, output_dir=output_dir))
