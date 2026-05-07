"""Research extension prototypes and analysis utilities."""

from research.adaptive import BitAllocation, LayerRankAllocation, allocate_adaptive_ranks, allocate_bits_from_importance
from research.dynamic import DynamicCompressionDecision, attention_entropy, dynamic_compression_policy
from research.hessian import diagonal_hessian_proxy, hessian_weighted_error
from research.query_aware import query_channel_importance, query_weighted_residual_norm

__all__ = [
    "BitAllocation",
    "DynamicCompressionDecision",
    "LayerRankAllocation",
    "allocate_adaptive_ranks",
    "allocate_bits_from_importance",
    "attention_entropy",
    "diagonal_hessian_proxy",
    "dynamic_compression_policy",
    "hessian_weighted_error",
    "query_channel_importance",
    "query_weighted_residual_norm",
]
