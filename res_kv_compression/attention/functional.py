"""Attention functions for baseline and reconstructed KV tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from compression.pipeline import HybridCompressedKVSnapshot
from models.kv_cache import KVCacheSnapshot, KVLayerCache

AttentionMode = Literal["baseline", "reconstructed", "low_rank_only", "hybrid"]


@dataclass(frozen=True)
class AttentionResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    output: torch.Tensor


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> AttentionResult:
    """Compute attention for tensors shaped ``[B, H, T, D]``."""

    _validate_attention_shapes(query, key, value)
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.einsum("bhqd,bhkd->bhqk", query, key) * scale
    if causal:
        logits = logits.masked_fill(_causal_mask(query.shape[-2], key.shape[-2], logits.device), float("-inf"))
    if attention_mask is not None:
        logits = logits + attention_mask
    probabilities = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    return AttentionResult(logits=logits, probabilities=probabilities, output=output)


def attention_for_mode(
    query: torch.Tensor,
    original: KVCacheSnapshot,
    compressed: HybridCompressedKVSnapshot,
    layer_idx: int,
    mode: AttentionMode,
    causal: bool = True,
) -> AttentionResult:
    """Run attention against baseline, reconstructed, or low-rank-only KV."""

    layer = select_layer_for_mode(original, compressed, layer_idx, mode)
    return scaled_dot_product_attention(query, layer.key, layer.value, causal=causal)


def select_layer_for_mode(
    original: KVCacheSnapshot,
    compressed: HybridCompressedKVSnapshot,
    layer_idx: int,
    mode: AttentionMode,
) -> KVLayerCache:
    if mode == "baseline":
        return _find_layer(original, layer_idx)
    if mode == "low_rank_only":
        return _find_layer(compressed.reconstruct(mode="low_rank_only"), layer_idx)
    if mode in {"reconstructed", "hybrid"}:
        return _find_layer(compressed.reconstruct(mode="reconstructed"), layer_idx)
    raise ValueError(f"Unsupported attention mode: {mode}")


def attention_logits(query: torch.Tensor, key: torch.Tensor, scale: bool = False) -> torch.Tensor:
    """Return ``QK^T`` logits, optionally scaled by ``sqrt(d)``."""

    _validate_query_key_shapes(query, key)
    logits = torch.einsum("bhqd,bhkd->bhqk", query, key)
    if scale:
        logits = logits / math.sqrt(query.shape[-1])
    return logits


def _find_layer(snapshot: KVCacheSnapshot, layer_idx: int) -> KVLayerCache:
    for layer in snapshot.layers:
        if layer.layer_idx == layer_idx:
            return layer
    raise ValueError(f"Layer {layer_idx} not found in KV snapshot")


def _causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    query_positions = torch.arange(query_len, device=device) + max(key_len - query_len, 0)
    key_positions = torch.arange(key_len, device=device)
    return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)


def _validate_attention_shapes(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    _validate_query_key_shapes(query, key)
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")


def _validate_query_key_shapes(query: torch.Tensor, key: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [B, H, T, D]")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch, head, and head_dim must match: {tuple(query.shape)} vs {tuple(key.shape)}")
