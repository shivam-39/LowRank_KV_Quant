"""Research objectives for low-rank plus quantized residual KV compression."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from attention.functional import attention_logits, scaled_dot_product_attention
from utils.config import ObjectiveConfig


@dataclass(frozen=True)
class ObjectiveBreakdown:
    reconstruction: float
    attention: float
    softmax: float
    output: float
    total: float


def reconstruction_loss(
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
) -> torch.Tensor:
    """Return ``||K-K_hat||_F^2 + ||V-V_hat||_F^2``."""

    return (key - key_hat).square().sum() + (value - value_hat).square().sum()


def attention_loss(query: torch.Tensor, key: torch.Tensor, key_hat: torch.Tensor) -> torch.Tensor:
    """Return ``||QK^T - QK_hat^T||_F^2`` without softmax scaling."""

    return (attention_logits(query, key) - attention_logits(query, key_hat)).square().sum()


def softmax_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    key_hat: torch.Tensor,
    loss_type: str = "kl",
) -> torch.Tensor:
    """Compare baseline and reconstructed attention distributions."""

    logits = attention_logits(query, key, scale=True)
    logits_hat = attention_logits(query, key_hat, scale=True)
    probabilities = torch.softmax(logits, dim=-1)
    probabilities_hat = torch.softmax(logits_hat, dim=-1)
    if loss_type == "mse":
        return (probabilities - probabilities_hat).square().sum()
    if loss_type == "kl":
        eps = torch.finfo(probabilities.dtype).eps
        return (
            probabilities.clamp_min(eps)
            * (probabilities.clamp_min(eps).log() - probabilities_hat.clamp_min(eps).log())
        ).sum()
    raise ValueError("loss_type must be kl or mse")


def output_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
) -> torch.Tensor:
    """Return ``||Attention(Q,K,V) - Attention(Q,K_hat,V_hat)||_F^2``."""

    baseline = scaled_dot_product_attention(query, key, value, causal=False)
    reconstructed = scaled_dot_product_attention(query, key_hat, value_hat, causal=False)
    return (baseline.output - reconstructed.output).square().sum()


def total_objective(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
    config: ObjectiveConfig,
) -> ObjectiveBreakdown:
    """Compute the configured weighted objective terms."""

    recon = reconstruction_loss(key, value, key_hat, value_hat)
    attn = attention_loss(query, key, key_hat) if config.use_attention_loss else torch.zeros_like(recon)
    sm = softmax_loss(query, key, key_hat, config.softmax_loss_type) if config.use_softmax_loss else torch.zeros_like(recon)
    out = output_loss(query, key, value, key_hat, value_hat) if config.use_output_loss else torch.zeros_like(recon)
    total = (
        config.lambda_recon * recon
        + config.lambda_attention * attn
        + config.lambda_softmax * sm
        + config.lambda_output * out
    )
    return ObjectiveBreakdown(
        reconstruction=_as_float(recon),
        attention=_as_float(attn),
        softmax=_as_float(sm),
        output=_as_float(out),
        total=_as_float(total),
    )


def perplexity_from_loss(loss: float) -> float:
    """Convert mean negative log-likelihood to perplexity."""

    return float(math.exp(loss))


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
