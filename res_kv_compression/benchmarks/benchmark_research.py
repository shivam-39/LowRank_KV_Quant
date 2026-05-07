"""Smoke benchmark for research extension helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kv_cache import KVCacheSnapshot, KVLayerCache
from research.adaptive import allocate_adaptive_ranks
from research.dynamic import attention_entropy, dynamic_compression_policy
from research.hessian import diagonal_hessian_proxy
from research.query_aware import query_channel_importance
from utils.config import CompressionConfig


def main() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 4, 16, 32)
    probabilities = torch.softmax(torch.randn(1, 4, 16, 64), dim=-1)
    snapshot = KVCacheSnapshot(
        layers=(
            KVLayerCache(
                layer_idx=0,
                key=torch.randn(1, 4, 64, 32),
                value=torch.randn(1, 4, 64, 32),
            ),
        )
    )
    ranks = allocate_adaptive_ranks(snapshot, CompressionConfig(rank=16, energy_threshold=0.95))
    entropy = attention_entropy(probabilities)
    decision = dynamic_compression_policy(context_length=64, attention_entropy_value=entropy, base_rank=16)

    print(
        json.dumps(
            {
                "benchmark": "research",
                "first_key_rank": ranks[0].key_head_ranks[0],
                "entropy": entropy,
                "dynamic_rank": decision.rank,
                "dynamic_bits": decision.quant_bits,
                "hessian_proxy_shape": list(diagonal_hessian_proxy(query).shape),
                "query_importance_sum": float(query_channel_importance(query).sum().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
