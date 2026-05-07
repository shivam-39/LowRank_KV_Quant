"""Smoke benchmark for fake residual quantization."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantization.fake_quant import fake_quantize
from utils.config import QuantizationConfig


def main() -> None:
    torch.manual_seed(0)
    residual = torch.randn(2, 8, 256, 64) * 0.05
    config = QuantizationConfig(quant_bits=4, group_size=32, per_channel=False, symmetric=True)

    start = time.perf_counter()
    quantized = fake_quantize(residual, config)
    dequantized = quantized.dequantize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    mse = torch.mean((residual - dequantized).square()).item()
    print(
        json.dumps(
            {
                "benchmark": "quantization",
                "elapsed_ms": elapsed_ms,
                "mse": mse,
                "num_bits": quantized.num_bits,
                "mode": quantized.scale_mode,
                "logical_bytes": quantized.logical_nbytes,
                "stored_bytes": quantized.stored_nbytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
