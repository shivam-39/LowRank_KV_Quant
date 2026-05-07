"""Dynamic compression policy prototypes for future streaming inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicCompressionDecision:
    rank: int
    quant_bits: int
    reason: str


def attention_entropy(probabilities: torch.Tensor) -> float:
    """Return mean normalized attention entropy in ``[0, 1]``."""

    if probabilities.ndim < 1:
        raise ValueError("probabilities must have at least one dimension")
    eps = torch.finfo(probabilities.dtype).eps
    entropy = -(probabilities.clamp_min(eps) * probabilities.clamp_min(eps).log()).sum(dim=-1)
    max_entropy = math.log(probabilities.shape[-1]) if probabilities.shape[-1] > 1 else 1.0
    return float((entropy / max_entropy).mean().item())


def dynamic_compression_policy(
    context_length: int,
    attention_entropy_value: float,
    base_rank: int,
    min_rank: int = 4,
) -> DynamicCompressionDecision:
    """Prototype token/context-adaptive rank and bit policy."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if base_rank <= 0 or min_rank <= 0:
        raise ValueError("ranks must be positive")
    if attention_entropy_value >= 0.75:
        return DynamicCompressionDecision(rank=base_rank, quant_bits=8, reason="high_entropy")
    if context_length >= 4096 and attention_entropy_value <= 0.35:
        return DynamicCompressionDecision(rank=max(min_rank, base_rank // 2), quant_bits=2, reason="long_low_entropy")
    return DynamicCompressionDecision(rank=max(min_rank, int(round(base_rank * 0.75))), quant_bits=4, reason="balanced")
