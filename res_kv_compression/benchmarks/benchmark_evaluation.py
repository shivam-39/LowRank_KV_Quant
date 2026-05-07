"""Smoke benchmark for evaluation utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.latency import benchmark_callable
from evaluation.long_context import make_token_windows
from evaluation.memory import snapshot_memory_report
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, QuantizationConfig


def main() -> None:
    torch.manual_seed(0)
    snapshot = KVCacheSnapshot(
        layers=(
            KVLayerCache(
                layer_idx=0,
                key=torch.randn(1, 4, 128, 32),
                value=torch.randn(1, 4, 128, 32),
            ),
        )
    )
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=8, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )
    memory = snapshot_memory_report(snapshot, compressed)
    latency = benchmark_callable(lambda: torch.mm(torch.randn(32, 32), torch.randn(32, 32)), warmup=1, iterations=3)
    windows = make_token_windows(torch.arange(1024), max_seq_len=256, stride=128)

    print(
        json.dumps(
            {
                "benchmark": "evaluation",
                "compression_ratio": memory.compression_ratio,
                "latency_mean_ms": latency.mean_ms,
                "num_long_context_windows": len(windows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
