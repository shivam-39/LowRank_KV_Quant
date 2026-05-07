"""Memory accounting for KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compression.pipeline import HybridCompressedKVSnapshot
from models.kv_cache import KVCacheSnapshot


@dataclass(frozen=True)
class MemoryReport:
    original_bytes: int
    compressed_logical_bytes: int | None = None
    compressed_stored_bytes: int | None = None

    @property
    def compression_ratio(self) -> float | None:
        if not self.compressed_logical_bytes:
            return None
        return self.original_bytes / self.compressed_logical_bytes

    @property
    def memory_savings(self) -> float | None:
        if not self.compressed_logical_bytes:
            return None
        return 1.0 - (self.compressed_logical_bytes / self.original_bytes)


def snapshot_memory_report(
    snapshot: KVCacheSnapshot,
    compressed: HybridCompressedKVSnapshot | None = None,
) -> MemoryReport:
    if compressed is None:
        return MemoryReport(original_bytes=snapshot.total_bytes)
    report = compressed.memory_report(snapshot)
    return MemoryReport(
        original_bytes=report.original_bytes,
        compressed_logical_bytes=report.total_logical_bytes,
        compressed_stored_bytes=report.total_stored_bytes,
    )


def estimate_hf_kv_cache_bytes(
    model_config: object,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> int:
    """Estimate dense HuggingFace KV cache bytes from model config fields."""

    num_layers = int(getattr(model_config, "num_hidden_layers"))
    num_attention_heads = int(getattr(model_config, "num_attention_heads"))
    num_key_value_heads = int(getattr(model_config, "num_key_value_heads", num_attention_heads))
    hidden_size = int(getattr(model_config, "hidden_size"))
    head_dim = int(getattr(model_config, "head_dim", hidden_size // num_attention_heads))
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * num_layers * batch_size * num_key_value_heads * seq_len * head_dim * element_size
