"""Smoke benchmark for low-rank KV tensor compression."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.low_rank import compress_tensor
from utils.config import CompressionConfig


def main() -> None:
    torch.manual_seed(0)
    left = torch.randn(2, 4, 128, 8)
    right = torch.randn(2, 4, 8, 32)
    tensor = torch.matmul(left, right)
    config = CompressionConfig(rank=8, decomposition_type="truncated_svd", granularity="per_head")

    start = time.perf_counter()
    approximation = compress_tensor(tensor, config)
    reconstruction = approximation.reconstruct()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(
        json.dumps(
            {
                "benchmark": "low_rank",
                "elapsed_ms": elapsed_ms,
                "rank": approximation.max_rank,
                "relative_error": approximation.relative_error(tensor),
                "compressed_bytes": approximation.factor_bytes,
                "original_bytes": tensor.numel() * tensor.element_size(),
                "reconstruction_shape": list(reconstruction.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
