"""Evaluation modules for perplexity, memory, latency, and attention quality."""

from evaluation.objectives import (
    ObjectiveBreakdown,
    attention_loss,
    output_loss,
    perplexity_from_loss,
    reconstruction_loss,
    softmax_loss,
    total_objective,
)
from evaluation.latency import LatencyResult, benchmark_callable
from evaluation.long_context import LongContextSummary, LongContextWindow, make_token_windows, summarize_long_context
from evaluation.memory import MemoryReport, estimate_hf_kv_cache_bytes, snapshot_memory_report
from evaluation.perplexity import PerplexityResult, evaluate_perplexity, load_wikitext_texts

__all__ = [
    "LatencyResult",
    "LongContextSummary",
    "LongContextWindow",
    "MemoryReport",
    "ObjectiveBreakdown",
    "PerplexityResult",
    "attention_loss",
    "benchmark_callable",
    "estimate_hf_kv_cache_bytes",
    "evaluate_perplexity",
    "load_wikitext_texts",
    "make_token_windows",
    "output_loss",
    "perplexity_from_loss",
    "reconstruction_loss",
    "snapshot_memory_report",
    "softmax_loss",
    "summarize_long_context",
    "total_objective",
]
