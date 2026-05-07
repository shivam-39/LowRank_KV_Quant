import torch

from compression.low_rank import compress_kv_snapshot
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from quantization.fake_quant import fake_quantize, qmax_for_bits, quantize_kv_residuals, quantize_residual
from utils.config import CompressionConfig, QuantizationConfig


def test_qmax_for_supported_bitwidths() -> None:
    assert qmax_for_bits(8) == 127
    assert qmax_for_bits(4) == 7
    assert qmax_for_bits(2) == 1


def test_per_tensor_int8_quantization_round_trips_scale_formula() -> None:
    tensor = torch.tensor([-1.0, 0.0, 1.0])
    config = QuantizationConfig(quant_bits=8, group_size=16, per_channel=False, symmetric=True)

    quantized = fake_quantize(tensor, config)

    assert quantized.scale_mode == "per_tensor"
    assert torch.equal(quantized.qvalues, torch.tensor([-127, 0, 127], dtype=torch.int8))
    assert torch.allclose(quantized.dequantize(), tensor)


def test_int4_quantization_clamps_to_signed_symmetric_range() -> None:
    tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    config = QuantizationConfig(quant_bits=4, group_size=16, per_channel=False, symmetric=True)

    quantized = fake_quantize(tensor, config)

    assert int(quantized.qvalues.min()) >= -7
    assert int(quantized.qvalues.max()) <= 7
    assert quantized.qmax == 7


def test_zero_tensor_quantization_is_stable() -> None:
    tensor = torch.zeros(2, 3)
    config = QuantizationConfig(quant_bits=4, group_size=16, per_channel=False, symmetric=True)

    quantized = fake_quantize(tensor, config)

    assert torch.isfinite(quantized.scales).all()
    assert torch.equal(quantized.dequantize(), tensor)


def test_per_channel_quantization_uses_last_dimension_scales() -> None:
    tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    config = QuantizationConfig(quant_bits=8, group_size=16, per_channel=True, symmetric=True)

    quantized = fake_quantize(tensor, config)

    assert quantized.scale_mode == "per_channel"
    assert quantized.scales.shape == (1, 2)
    error = (quantized.dequantize() - tensor).abs()
    assert error.max() <= quantized.scales.max() / 2 + 1e-6


def test_group_quantization_pads_and_restores_shape() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(2, 7)
    config = QuantizationConfig(quant_bits=4, group_size=3, per_channel=False, symmetric=True)

    quantized = fake_quantize(tensor, config)
    dequantized = quantized.dequantize()

    assert quantized.scale_mode == "group"
    assert quantized.qvalues.shape[-1] == 9
    assert dequantized.shape == tensor.shape


def test_quantize_residual_matches_original_minus_low_rank() -> None:
    original = torch.tensor([1.0, 2.0, 4.0])
    low_rank = torch.tensor([0.5, 1.0, 2.0])
    config = QuantizationConfig(quant_bits=8, group_size=16, per_channel=False, symmetric=True)

    residual = quantize_residual(original, low_rank, config).dequantize()

    target = original - low_rank
    assert (residual - target).abs().max() <= residual.abs().max() / 127 + 1e-6


def test_quantize_kv_residuals_dequantizes_to_residual_snapshot() -> None:
    torch.manual_seed(0)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)
    snapshot = KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))
    low_rank = compress_kv_snapshot(snapshot, CompressionConfig(rank=2, granularity="per_head"))
    config = QuantizationConfig(quant_bits=4, group_size=16, per_channel=False, symmetric=True)

    quantized = quantize_kv_residuals(snapshot, low_rank, config)
    residual_snapshot = quantized.dequantize()

    assert residual_snapshot.layers[0].key.shape == key.shape
    assert residual_snapshot.layers[0].value.shape == value.shape
