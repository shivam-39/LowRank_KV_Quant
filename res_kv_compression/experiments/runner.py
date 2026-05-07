"""Experiment runner boundary.

The runner is intentionally thin in Phase 1. Later phases register concrete
compression and evaluation components behind this stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.latency import benchmark_callable
from evaluation.memory import estimate_hf_kv_cache_bytes, snapshot_memory_report
from evaluation.perplexity import evaluate_perplexity, load_wikitext_texts
from models.kv_cache import KVCacheExtractor
from models.loader import load_causal_lm_and_tokenizer, resolve_torch_dtype
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
    metrics: dict[str, float] = {}
    artifacts = {"output_dir": str(config.logging.output_dir)}
    if model_backed_eval:
        logger.info("Loading model for configured evaluation")
        loaded_model = load_causal_lm_and_tokenizer(config.model)
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        model_device = next(model.parameters()).device

        if config.evaluation.perplexity_eval:
            texts = load_wikitext_texts(
                config.evaluation.dataset_name,
                config.evaluation.dataset_config,
                config.evaluation.split,
                config.evaluation.max_eval_samples,
            )
            ppl = evaluate_perplexity(
                model,
                tokenizer,
                texts,
                max_seq_len=config.model.max_seq_len,
                stride=config.evaluation.stride,
                device=model_device,
            )
            metrics["perplexity"] = ppl.perplexity
            metrics["nll"] = ppl.nll
            metrics["perplexity_tokens"] = float(ppl.num_tokens)

        prompt_inputs = tokenizer(config.evaluation.prompt, return_tensors="pt").to(model_device)
        if config.evaluation.latency_eval:
            latency = benchmark_callable(
                lambda: model(**prompt_inputs, use_cache=True),
                warmup=config.evaluation.latency_warmup,
                iterations=config.evaluation.latency_iters,
                tokens_per_iteration=int(prompt_inputs["input_ids"].numel()),
            )
            metrics["latency_mean_ms"] = latency.mean_ms
            metrics["latency_p50_ms"] = latency.p50_ms
            metrics["latency_p95_ms"] = latency.p95_ms
            if latency.tokens_per_sec is not None:
                metrics["tokens_per_sec"] = latency.tokens_per_sec

        if config.evaluation.memory_eval:
            dtype = resolve_torch_dtype(config.model.dtype)
            dense_estimate = estimate_hf_kv_cache_bytes(
                model.config,
                batch_size=config.evaluation.batch_size,
                seq_len=config.model.max_seq_len,
                dtype=dtype,
            )
            snapshot = KVCacheExtractor().extract_from_inputs(
                model,
                prompt_inputs["input_ids"],
                attention_mask=prompt_inputs.get("attention_mask"),
                metadata={"prompt": config.evaluation.prompt},
            )
            compressed = compress_kv_snapshot_hybrid(snapshot, config.compression, config.quantization)
            memory = snapshot_memory_report(snapshot, compressed)
            metrics["estimated_dense_kv_bytes"] = float(dense_estimate)
            metrics["prompt_kv_bytes"] = float(memory.original_bytes)
            if memory.compressed_logical_bytes is not None:
                metrics["compressed_kv_logical_bytes"] = float(memory.compressed_logical_bytes)
            if memory.compression_ratio is not None:
                metrics["memory_compression_ratio"] = memory.compression_ratio
            if memory.memory_savings is not None:
                metrics["memory_savings"] = memory.memory_savings
            metrics["prompt_reconstruction_error"] = compressed.reconstruction_error(snapshot)

        if config.evaluation.long_context_eval:
            encoded = tokenizer(config.evaluation.prompt, return_tensors="pt")["input_ids"][0]
            metrics["long_context_tokens"] = float(encoded.numel())
            metrics["long_context_enabled"] = 1.0

        metadata["status"] = "evaluation_complete"
    else:
        metadata["status"] = "setup_complete"

    result = ExperimentResult(
        experiment=config.experiment.name,
        metrics=metrics,
        artifacts=artifacts,
        metadata=metadata,
    )
    experiment_logger.log_metrics(metrics or {"setup_complete": 1.0})
    return result
