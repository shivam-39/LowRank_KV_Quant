import torch

from compression.low_rank import (
    compress_kv_snapshot,
    compress_matrix,
    compress_tensor,
    select_rank_by_energy,
)
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig


def test_select_rank_by_energy_uses_squared_singular_values() -> None:
    singular_values = torch.tensor([3.0, 1.0, 1.0])

    assert select_rank_by_energy(singular_values, 0.80) == 1
    assert select_rank_by_energy(singular_values, 0.95) == 3


def test_full_rank_truncated_svd_reconstructs_matrix() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(8, 5)

    approximation = compress_matrix(matrix, rank=5)
    reconstruction = approximation.reconstruct()

    assert approximation.rank == 5
    assert torch.allclose(reconstruction, matrix, atol=1e-5)


def test_half_precision_matrix_uses_float32_decomposition() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(8, 5, dtype=torch.float16)

    approximation = compress_matrix(matrix, rank=5)
    reconstruction = approximation.reconstruct()

    assert approximation.u.dtype == torch.float32
    assert reconstruction.dtype == torch.float32
    assert torch.allclose(reconstruction, matrix.float(), atol=5e-3)


def test_adaptive_rank_selects_smaller_rank_for_low_rank_matrix() -> None:
    torch.manual_seed(0)
    left = torch.randn(16, 2)
    right = torch.randn(2, 8)
    matrix = left @ right

    approximation = compress_matrix(matrix, rank=8, adaptive_rank=True, energy_threshold=0.999)

    assert approximation.rank <= 2
    assert approximation.metrics(matrix).relative_error < 1e-5


def test_randomized_svd_returns_requested_rank_shape() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(32, 16)
    approximation = compress_matrix(
        matrix,
        rank=4,
        decomposition_type="randomized_svd",
        randomized_oversamples=4,
        randomized_n_iter=1,
    )

    assert approximation.u.shape == (32, 4)
    assert approximation.s.shape == (4,)
    assert approximation.vh.shape == (4, 16)


def test_per_head_tensor_compression_reconstructs_shape() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(2, 3, 6, 4)
    config = CompressionConfig(rank=4, granularity="per_head")

    approximation = compress_tensor(tensor, config)
    reconstruction = approximation.reconstruct()

    assert approximation.max_rank == 4
    assert reconstruction.shape == tensor.shape
    assert torch.allclose(reconstruction, tensor, atol=1e-5)


def test_compress_kv_snapshot_round_trips_shapes() -> None:
    torch.manual_seed(0)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)
    snapshot = KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))
    config = CompressionConfig(rank=2, granularity="per_head")

    low_rank = compress_kv_snapshot(snapshot, config)
    reconstructed = low_rank.reconstruct()

    assert reconstructed.num_layers == 1
    assert reconstructed.layers[0].key.shape == key.shape
    assert reconstructed.layers[0].value.shape == value.shape
