import torch

from attention.functional import attention_for_mode, scaled_dot_product_attention
from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.objectives import (
    attention_loss,
    output_loss,
    reconstruction_loss,
    softmax_loss,
    total_objective,
)
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, ObjectiveConfig, QuantizationConfig


def test_scaled_dot_product_attention_matches_manual_computation() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])

    result = scaled_dot_product_attention(query, key, value, causal=False)
    expected_probs = torch.softmax(torch.tensor([1.0 / 2**0.5, 0.0]), dim=0)
    expected_output = expected_probs[0] * value[0, 0, 0] + expected_probs[1] * value[0, 0, 1]

    assert torch.allclose(result.probabilities[0, 0, 0], expected_probs)
    assert torch.allclose(result.output[0, 0, 0], expected_output)


def test_causal_mask_blocks_future_keys() -> None:
    query = torch.ones(1, 1, 2, 1)
    key = torch.ones(1, 1, 2, 1)
    value = torch.tensor([[[[1.0], [3.0]]]])

    result = scaled_dot_product_attention(query, key, value, causal=True)

    assert torch.allclose(result.probabilities[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.output[0, 0, 0], torch.tensor([1.0]))


def test_attention_for_mode_uses_reconstructed_kv() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    snapshot = KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=4, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    baseline = attention_for_mode(query, snapshot, compressed, 0, mode="baseline", causal=False)
    reconstructed = attention_for_mode(query, snapshot, compressed, 0, mode="reconstructed", causal=False)

    assert torch.allclose(baseline.output, reconstructed.output, atol=1e-5)


def test_objective_losses_are_zero_for_identical_tensors() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    assert reconstruction_loss(key, value, key, value).item() == 0.0
    assert attention_loss(query, key, key).item() == 0.0
    assert softmax_loss(query, key, key).abs().item() < 1e-6
    assert output_loss(query, key, value, key, value).item() == 0.0


def test_total_objective_respects_disabled_terms_and_lambdas() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 1, 2, 3)
    key = torch.randn(1, 1, 4, 3)
    value = torch.randn(1, 1, 4, 3)
    key_hat = key + 0.1
    value_hat = value - 0.2
    config = ObjectiveConfig(
        use_attention_loss=False,
        use_softmax_loss=False,
        use_output_loss=False,
        lambda_recon=2.0,
    )

    result = total_objective(query, key, value, key_hat, value_hat, config)

    assert result.attention == 0.0
    assert result.softmax == 0.0
    assert result.output == 0.0
    assert abs(result.total - 2.0 * result.reconstruction) < 1e-6
