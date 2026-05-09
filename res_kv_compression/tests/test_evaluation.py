import math
from types import SimpleNamespace

import torch

from compression.pipeline import compress_kv_snapshot_hybrid
from evaluation.generation_memory import evaluate_compressed_generation_memory
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


class ToyGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, cache_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.cache_dtype = cache_dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        batch_size, input_len = input_ids.shape
        new_key = input_ids.to(self.cache_dtype).reshape(batch_size, 1, input_len, 1).repeat(1, 1, 1, 2)
        new_value = (input_ids.to(self.cache_dtype) + 0.5).reshape(batch_size, 1, input_len, 1).repeat(1, 1, 1, 2)
        if past_key_values is None:
            key = new_key
            value = new_value
        else:
            assert past_key_values[0][0].dtype == self.cache_dtype
            assert past_key_values[0][1].dtype == self.cache_dtype
            key = torch.cat((past_key_values[0][0], new_key), dim=2)
            value = torch.cat((past_key_values[0][1], new_value), dim=2)

        logits = torch.zeros(batch_size, input_len, self.vocab_size)
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


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


def test_compressed_generation_memory_eval_records_growth(tmp_path) -> None:
    result = evaluate_compressed_generation_memory(
        model=ToyGenerationModel(cache_dtype=torch.float16),
        tokenizer=ToyTokenizer(),
        prompt="one two",
        max_seq_len=5,
        compression_config=CompressionConfig(rank=1, granularity="per_head"),
        quantization_config=QuantizationConfig(quant_bits=4, group_size=16),
        output_dir=tmp_path,
    )

    assert result.generated_tokens == 3
    assert [point.sequence_length for point in result.points] == [2, 3, 4, 5]
    assert result.points[-1].dense_kv_bytes > result.points[0].dense_kv_bytes
    assert result.points[-1].compressed_kv_logical_bytes > 0
    assert result.trace_path.exists()


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
