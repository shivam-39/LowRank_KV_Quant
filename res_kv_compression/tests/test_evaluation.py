import math
from types import SimpleNamespace

import torch

from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.latency import benchmark_callable
from evaluation.long_context import make_token_windows, summarize_long_context
from evaluation.memory import estimate_hf_kv_cache_bytes, snapshot_memory_report
from evaluation.perplexity import evaluate_perplexity
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, QuantizationConfig


class ToyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del return_tensors
        token_ids = torch.arange(1, len(text.split()) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": token_ids}


class ToyLossModel(torch.nn.Module):
    def __init__(self, loss: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.loss = loss

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> SimpleNamespace:
        del input_ids, labels
        return SimpleNamespace(loss=torch.tensor(self.loss))


def test_evaluate_perplexity_with_constant_loss_model() -> None:
    result = evaluate_perplexity(
        ToyLossModel(math.log(2.0)),
        ToyTokenizer(),
        ["one two three four"],
        max_seq_len=4,
        stride=4,
    )

    assert abs(result.perplexity - 2.0) < 1e-6
    assert result.num_tokens == 4


def test_snapshot_memory_report_uses_hybrid_logical_bytes() -> None:
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
    compressed = compress_kv_snapshot_hybrid(
        snapshot,
        CompressionConfig(rank=2, granularity="per_head"),
        QuantizationConfig(quant_bits=4, group_size=16),
    )

    report = snapshot_memory_report(snapshot, compressed)

    assert report.original_bytes == snapshot.total_bytes
    assert report.compressed_logical_bytes is not None
    assert report.compression_ratio is not None


def test_estimate_hf_kv_cache_bytes() -> None:
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
    )

    assert estimate_hf_kv_cache_bytes(config, batch_size=1, seq_len=8, dtype=torch.float16) == 512


def test_benchmark_callable_reports_latency() -> None:
    result = benchmark_callable(lambda: 1 + 1, warmup=0, iterations=2, tokens_per_iteration=4)

    assert result.iterations == 2
    assert result.mean_ms >= 0.0
    assert result.tokens_per_sec is not None


def test_long_context_windowing() -> None:
    token_ids = torch.arange(10)

    windows = make_token_windows(token_ids, max_seq_len=4, stride=3)
    summary = summarize_long_context(token_ids, max_seq_len=4, stride=3)

    assert [(window.start, window.end) for window in windows] == [(0, 4), (3, 7), (6, 10)]
    assert summary.num_windows == 3
    assert summary.max_window_length == 4
