"""Residual quantization modules."""

from quantization.fake_quant import (
    QuantizationMetrics,
    QuantizedKVLayerResidual,
    QuantizedKVResidualSnapshot,
    QuantizedTensor,
    fake_quantize,
    qmax_for_bits,
    quantize_kv_residuals,
    quantize_residual,
)

__all__ = [
    "QuantizationMetrics",
    "QuantizedKVLayerResidual",
    "QuantizedKVResidualSnapshot",
    "QuantizedTensor",
    "fake_quantize",
    "qmax_for_bits",
    "quantize_kv_residuals",
    "quantize_residual",
]
