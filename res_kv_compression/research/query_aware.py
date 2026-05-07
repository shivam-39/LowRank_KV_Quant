"""Query-aware compression analysis helpers."""

from __future__ import annotations

import torch


def query_channel_importance(query: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Return normalized per-channel query importance."""

    if query.ndim != 4:
        raise ValueError("query must have shape [B, H, T, D]")
    if p not in {1, 2}:
        raise ValueError("p must be 1 or 2")
    scores = query.abs().mean(dim=(0, 1, 2)) if p == 1 else query.square().mean(dim=(0, 1, 2))
    total = scores.sum()
    if total <= 0:
        return torch.full_like(scores, 1.0 / scores.numel())
    return scores / total


def query_weighted_residual_norm(residual: torch.Tensor, query_importance: torch.Tensor) -> torch.Tensor:
    """Weight residual channels by observed query importance."""

    if residual.shape[-1] != query_importance.shape[-1]:
        raise ValueError("residual last dimension must match query_importance")
    weights = query_importance.reshape(*((1,) * (residual.ndim - 1)), -1)
    return (residual.square() * weights).sum()
