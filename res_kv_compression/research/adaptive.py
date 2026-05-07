"""Adaptive rank and bit allocation research helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compression.low_rank import select_rank_by_energy
from models.kv_cache import KVCacheSnapshot
from utils.config import CompressionConfig


@dataclass(frozen=True)
class LayerRankAllocation:
    layer_idx: int
    key_head_ranks: tuple[int, ...]
    value_head_ranks: tuple[int, ...]

    @property
    def max_rank(self) -> int:
        return max((*self.key_head_ranks, *self.value_head_ranks))


@dataclass(frozen=True)
class BitAllocation:
    layer_idx: int
    key_bits: int
    value_bits: int


def allocate_adaptive_ranks(
    snapshot: KVCacheSnapshot,
    config: CompressionConfig,
) -> tuple[LayerRankAllocation, ...]:
    """Allocate per-head ranks from singular-value energy thresholds."""

    allocations: list[LayerRankAllocation] = []
    for layer in snapshot.layers:
        key_ranks = tuple(
            _rank_for_head(layer.key[:, head_idx, :, :], config.rank, config.energy_threshold)
            for head_idx in range(layer.key.shape[1])
        )
        value_ranks = tuple(
            _rank_for_head(layer.value[:, head_idx, :, :], config.rank, config.energy_threshold)
            for head_idx in range(layer.value.shape[1])
        )
        allocations.append(
            LayerRankAllocation(
                layer_idx=layer.layer_idx,
                key_head_ranks=key_ranks,
                value_head_ranks=value_ranks,
            )
        )
    return tuple(allocations)


def allocate_bits_from_importance(
    importance: torch.Tensor,
    low_bits: int = 2,
    mid_bits: int = 4,
    high_bits: int = 8,
    high_threshold: float = 0.75,
    mid_threshold: float = 0.25,
) -> torch.Tensor:
    """Map normalized channel/head importance scores to bitwidths."""

    if low_bits not in {2, 4, 8} or mid_bits not in {2, 4, 8} or high_bits not in {2, 4, 8}:
        raise ValueError("bit options must be in {2, 4, 8}")
    if importance.numel() == 0:
        raise ValueError("importance must be non-empty")
    normalized = importance.to(torch.float32)
    if normalized.max() > 0:
        normalized = normalized / normalized.max()
    bits = torch.full_like(normalized, low_bits, dtype=torch.int64)
    bits = torch.where(normalized >= mid_threshold, torch.full_like(bits, mid_bits), bits)
    bits = torch.where(normalized >= high_threshold, torch.full_like(bits, high_bits), bits)
    return bits


def _rank_for_head(head_tensor: torch.Tensor, max_rank: int, threshold: float) -> int:
    matrix = head_tensor.reshape(-1, head_tensor.shape[-1])
    singular_values = torch.linalg.svdvals(matrix)
    return min(select_rank_by_energy(singular_values, threshold), max_rank)
