"""Smoke benchmark for baseline and reconstructed attention."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention.functional import attention_for_mode
from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.objectives import total_objective
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, ObjectiveConfig, QuantizationConfig


def main() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 4, 8, 32)
    key = torch.randn(1, 4, 128, 32)
    value = torch.randn(1, 4, 128, 32)
    snapshot = KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=8, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    start = time.perf_counter()
    baseline = attention_for_mode(query, snapshot, compressed, 0, mode="baseline")
    reconstructed = attention_for_mode(query, snapshot, compressed, 0, mode="reconstructed")
    objective = total_objective(
        query,
        key,
        value,
        compressed.reconstruct().layers[0].key,
        compressed.reconstruct().layers[0].value,
        ObjectiveConfig(),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(
        json.dumps(
            {
                "benchmark": "attention",
                "elapsed_ms": elapsed_ms,
                "output_mse": float(torch.mean((baseline.output - reconstructed.output).square()).item()),
                "total_loss": objective.total,
                "attention_loss": objective.attention,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
