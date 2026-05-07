"""Smoke benchmark for KV cache extraction on a toy causal LM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kv_cache import KVCacheExtractor


class ToyCausalLM(torch.nn.Module):
    def __init__(self, num_layers: int = 4, num_heads: int = 8, head_dim: int = 16) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        batch_size, seq_len = input_ids.shape
        base = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim)
        past_key_values = tuple((base + layer_idx, base - layer_idx) for layer_idx in range(self.num_layers))
        return SimpleNamespace(past_key_values=past_key_values)


def main() -> None:
    model = ToyCausalLM()
    input_ids = torch.ones(2, 128, dtype=torch.long)

    start = time.perf_counter()
    snapshot = KVCacheExtractor().extract_from_inputs(model, input_ids)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(
        json.dumps(
            {
                "benchmark": "kv_extraction",
                "elapsed_ms": elapsed_ms,
                "num_layers": snapshot.num_layers,
                "total_tokens": snapshot.total_tokens,
                "total_bytes": snapshot.total_bytes,
                "layer0_key_shape": list(snapshot.layers[0].key.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
