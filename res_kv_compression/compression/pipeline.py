"""End-to-end low-rank plus quantized residual KV compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from compression.low_rank import LowRankKVSnapshot, compress_kv_snapshot
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from quantization.fake_quant import QuantizedKVResidualSnapshot, quantize_kv_residuals
from utils.config import CompressionConfig, QuantizationConfig

ReconstructionMode = Literal["reconstructed", "low_rank_only", "hybrid"]


@dataclass(frozen=True)
class CompressionMemoryReport:
    original_bytes: int
    low_rank_bytes: int
    residual_logical_bytes: int
    residual_stored_bytes: int

    @property
    def total_logical_bytes(self) -> int:
        return self.low_rank_bytes + self.residual_logical_bytes

    @property
    def total_stored_bytes(self) -> int:
        return self.low_rank_bytes + self.residual_stored_bytes

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.total_logical_bytes if self.total_logical_bytes > 0 else float("inf")


@dataclass(frozen=True)
class HybridCompressedKVSnapshot:
    """Compressed KV snapshot storing low-rank factors and quantized residuals."""

    low_rank: LowRankKVSnapshot
    residual: QuantizedKVResidualSnapshot

    @property
    def num_layers(self) -> int:
        return len(self.low_rank.layers)

    def reconstruct(self, mode: ReconstructionMode = "reconstructed") -> KVCacheSnapshot:
        """Reconstruct KV tensors according to the requested attention mode."""

        if mode == "low_rank_only":
            return self.low_rank.reconstruct()
        if mode not in {"reconstructed", "hybrid"}:
            raise ValueError("mode must be reconstructed, low_rank_only, or hybrid")

        low_rank_snapshot = self.low_rank.reconstruct()
        residual_snapshot = self.residual.dequantize()
        if low_rank_snapshot.num_layers != residual_snapshot.num_layers:
            raise ValueError("low-rank and residual snapshots must contain the same number of layers")

        layers: list[KVLayerCache] = []
        for low_rank_layer, residual_layer in zip(
            low_rank_snapshot.layers,
            residual_snapshot.layers,
            strict=True,
        ):
            layers.append(
                KVLayerCache(
                    layer_idx=low_rank_layer.layer_idx,
                    key=low_rank_layer.key + residual_layer.key,
                    value=low_rank_layer.value + residual_layer.value,
                )
            )
        return KVCacheSnapshot(layers=tuple(layers), metadata={"source": "hybrid_reconstruction"})

    def memory_report(self, original: KVCacheSnapshot) -> CompressionMemoryReport:
        low_rank_bytes = sum(
            layer.key.factor_bytes + layer.value.factor_bytes
            for layer in self.low_rank.layers
        )
        residual_logical_bytes = sum(
            layer.key_residual.logical_nbytes + layer.value_residual.logical_nbytes
            for layer in self.residual.layers
        )
        residual_stored_bytes = sum(
            layer.key_residual.stored_nbytes + layer.value_residual.stored_nbytes
            for layer in self.residual.layers
        )
        return CompressionMemoryReport(
            original_bytes=original.total_bytes,
            low_rank_bytes=low_rank_bytes,
            residual_logical_bytes=residual_logical_bytes,
            residual_stored_bytes=residual_stored_bytes,
        )

    def reconstruction_error(self, original: KVCacheSnapshot, mode: ReconstructionMode = "reconstructed") -> float:
        reconstructed = self.reconstruct(mode)
        squared_error = torch.zeros((), dtype=torch.float64)
        squared_norm = torch.zeros((), dtype=torch.float64)
        for original_layer, reconstructed_layer in zip(original.layers, reconstructed.layers, strict=True):
            squared_error = squared_error + (original_layer.key - reconstructed_layer.key).double().square().sum()
            squared_error = squared_error + (original_layer.value - reconstructed_layer.value).double().square().sum()
            squared_norm = squared_norm + original_layer.key.double().square().sum()
            squared_norm = squared_norm + original_layer.value.double().square().sum()
        if squared_norm <= 0:
            return float(torch.sqrt(squared_error).item())
        return float(torch.sqrt(squared_error / squared_norm).item())


def compress_kv_snapshot_hybrid(
    snapshot: KVCacheSnapshot,
    compression_config: CompressionConfig,
    quantization_config: QuantizationConfig,
) -> HybridCompressedKVSnapshot:
    """Apply ``K,V ~= low_rank + quantized_residual`` to a KV snapshot."""

    low_rank = compress_kv_snapshot(snapshot, compression_config)
    residual = quantize_kv_residuals(snapshot, low_rank, quantization_config)
    return HybridCompressedKVSnapshot(low_rank=low_rank, residual=residual)
