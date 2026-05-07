import torch

from models.kv_cache import KVCacheSnapshot, KVLayerCache
from research.adaptive import allocate_adaptive_ranks, allocate_bits_from_importance
from research.dynamic import attention_entropy, dynamic_compression_policy
from research.hessian import diagonal_hessian_proxy, hessian_weighted_error
from research.query_aware import query_channel_importance, query_weighted_residual_norm
from utils.config import CompressionConfig


def test_allocate_adaptive_ranks_returns_per_head_allocations() -> None:
    torch.manual_seed(0)
    snapshot = KVCacheSnapshot(
        layers=(
            KVLayerCache(
                layer_idx=0,
                key=torch.randn(1, 2, 8, 4),
                value=torch.randn(1, 2, 8, 4),
            ),
        )
    )

    allocations = allocate_adaptive_ranks(snapshot, CompressionConfig(rank=4, energy_threshold=0.9))

    assert allocations[0].layer_idx == 0
    assert len(allocations[0].key_head_ranks) == 2
    assert allocations[0].max_rank <= 4


def test_allocate_bits_from_importance_maps_thresholds() -> None:
    bits = allocate_bits_from_importance(torch.tensor([0.1, 0.5, 1.0]))

    assert torch.equal(bits, torch.tensor([2, 4, 8]))


def test_hessian_proxy_and_weighted_error() -> None:
    query = torch.ones(1, 2, 3, 4)
    error = torch.ones(1, 2, 3, 4)
    hessian = diagonal_hessian_proxy(query)

    assert torch.equal(hessian, torch.ones(4))
    assert hessian_weighted_error(error, hessian).item() == 24.0


def test_query_importance_is_normalized_and_weights_residuals() -> None:
    query = torch.zeros(1, 1, 2, 3)
    query[..., 0] = 2.0
    importance = query_channel_importance(query)
    residual = torch.ones(1, 1, 2, 3)

    assert torch.allclose(importance.sum(), torch.tensor(1.0))
    assert importance[0] > importance[1]
    assert query_weighted_residual_norm(residual, importance).item() == 2.0


def test_attention_entropy_and_dynamic_policy() -> None:
    uniform = torch.full((1, 1, 2, 4), 0.25)
    entropy = attention_entropy(uniform)
    high = dynamic_compression_policy(context_length=128, attention_entropy_value=entropy, base_rank=16)
    low = dynamic_compression_policy(context_length=8192, attention_entropy_value=0.1, base_rank=16)

    assert abs(entropy - 1.0) < 1e-6
    assert high.quant_bits == 8
    assert low.quant_bits == 2
    assert low.rank == 8
