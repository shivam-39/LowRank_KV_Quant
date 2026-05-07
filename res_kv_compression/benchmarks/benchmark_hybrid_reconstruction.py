"""Smoke benchmark for hybrid low-rank plus quantized residual reconstruction."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.pipeline import compress_kv_snapshot_hybrid
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, QuantizationConfig


def main() -> None:
    torch.manual_seed(0)
    key = torch.randn(1, 8, 256, 64)
    value = torch.randn(1, 8, 256, 64)
    snapshot = KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))
    compression = CompressionConfig(rank=16, granularity="per_head")
    quantization = QuantizationConfig(quant_bits=4, group_size=32)

    start = time.perf_counter()
    compressed = compress_kv_snapshot_hybrid(snapshot, compression, quantization)
    reconstructed = compressed.reconstruct()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    report = compressed.memory_report(snapshot)

    error = torch.linalg.vector_norm(reconstructed.layers[0].key - key) / torch.linalg.vector_norm(key)
    print(
        json.dumps(
            {
                "benchmark": "hybrid_reconstruction",
                "elapsed_ms": elapsed_ms,
                "key_relative_error": float(error.item()),
                "compression_ratio": report.compression_ratio,
                "logical_compressed_bytes": report.total_logical_bytes,
                "original_bytes": report.original_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
