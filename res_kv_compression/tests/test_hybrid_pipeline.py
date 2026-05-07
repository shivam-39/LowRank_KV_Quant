import torch

from compression.pipeline import compress_kv_snapshot_hybrid
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, QuantizationConfig


def _snapshot() -> KVCacheSnapshot:
    torch.manual_seed(0)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)
    return KVCacheSnapshot(layers=(KVLayerCache(layer_idx=0, key=key, value=value),))


def test_hybrid_reconstruction_preserves_shapes() -> None:
    snapshot = _snapshot()
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=2, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    reconstructed = compressed.reconstruct()

    assert reconstructed.num_layers == snapshot.num_layers
    assert reconstructed.layers[0].key.shape == snapshot.layers[0].key.shape
    assert reconstructed.layers[0].value.shape == snapshot.layers[0].value.shape


def test_full_rank_low_rank_only_reconstruction_is_exact() -> None:
    snapshot = _snapshot()
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=4, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    low_rank_only = compressed.reconstruct(mode="low_rank_only")

    assert torch.allclose(low_rank_only.layers[0].key, snapshot.layers[0].key, atol=1e-5)
    assert torch.allclose(low_rank_only.layers[0].value, snapshot.layers[0].value, atol=1e-5)


def test_hybrid_error_is_no_worse_than_low_rank_only_for_residual_quantization() -> None:
    snapshot = _snapshot()
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=1, granularity="per_head"),
        QuantizationConfig(quant_bits=8, group_size=16),
    )

    assert compressed.reconstruction_error(snapshot, mode="reconstructed") <= (
        compressed.reconstruction_error(snapshot, mode="low_rank_only") + 1e-6
    )


def test_memory_report_tracks_logical_quantized_bytes() -> None:
    snapshot = _snapshot()
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=1, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    report = compressed.memory_report(snapshot)

    assert report.original_bytes == snapshot.total_bytes
    assert report.low_rank_bytes > 0
    assert report.residual_logical_bytes > 0
    assert report.total_logical_bytes == report.low_rank_bytes + report.residual_logical_bytes
