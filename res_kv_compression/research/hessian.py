"""Hessian-aware compression proxies inspired by GPTQ-style sensitivity."""

from __future__ import annotations

import torch


def diagonal_hessian_proxy(query: torch.Tensor) -> torch.Tensor:
    """Approximate key-channel sensitivity with ``E[Q^2]`` over batch/head/time."""

    if query.ndim != 4:
        raise ValueError("query must have shape [B, H, T, D]")
    return query.square().mean(dim=(0, 1, 2))


def hessian_weighted_error(error: torch.Tensor, hessian_diag: torch.Tensor) -> torch.Tensor:
    """Return ``sum_d H_d * error_d^2`` with broadcast over leading dimensions."""

    if error.shape[-1] != hessian_diag.shape[-1]:
        raise ValueError("error last dimension must match hessian_diag")
    return (error.square() * hessian_diag.reshape(*((1,) * (error.ndim - 1)), -1)).sum()
