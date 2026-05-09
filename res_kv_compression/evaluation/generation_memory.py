"""Autoregressive generation with compressed KV cache memory tracing."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from compression.pipeline import compress_kv_snapshot_hybrid
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig, QuantizationConfig


@dataclass(frozen=True)
class GenerationMemoryPoint:
    sequence_length: int
    dense_kv_bytes: int
    compressed_kv_logical_bytes: int
    compressed_kv_stored_bytes: int
    compression_ratio: float
    memory_savings: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "generation_seq_len": float(self.sequence_length),
            "generation_dense_kv_bytes": float(self.dense_kv_bytes),
            "generation_compressed_kv_logical_bytes": float(self.compressed_kv_logical_bytes),
            "generation_compressed_kv_stored_bytes": float(self.compressed_kv_stored_bytes),
            "generation_memory_compression_ratio": self.compression_ratio,
            "generation_memory_savings": self.memory_savings,
        }


@dataclass(frozen=True)
class GenerationMemoryResult:
    points: tuple[GenerationMemoryPoint, ...]
    generated_tokens: int
    trace_path: Path
    plot_path: Path | None = None

    @property
    def final_point(self) -> GenerationMemoryPoint:
        return self.points[-1]


@torch.no_grad()
def evaluate_compressed_generation_memory(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_seq_len: int,
    compression_config: CompressionConfig,
    quantization_config: QuantizationConfig,
    output_dir: str | Path,
) -> GenerationMemoryResult:
    """Generate until ``max_seq_len`` using reconstructed compressed KV caches."""

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    model.eval()
    device = next(model.parameters()).device
    encoded = _tokenize_prompt(tokenizer, prompt, max_seq_len)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = _attention_mask(encoded, input_ids).to(device)
    prompt_len = int(input_ids.shape[-1])
    if prompt_len <= 0:
        raise ValueError("prompt must tokenize to at least one token")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = _require_past_key_values(outputs)
    next_token = _greedy_next_token(outputs)
    sequence_length = prompt_len

    points: list[GenerationMemoryPoint] = []
    while sequence_length < max_seq_len:
        point, reconstructed_cache = _compress_and_record_cache(
            past_key_values,
            compression_config,
            quantization_config,
        )
        points.append(point)

        decode_attention_mask = torch.ones(
            input_ids.shape[0],
            sequence_length + 1,
            dtype=attention_mask.dtype,
            device=device,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=decode_attention_mask,
            past_key_values=reconstructed_cache,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = _require_past_key_values(outputs)
        next_token = _greedy_next_token(outputs)
        sequence_length += 1

    if not points or points[-1].sequence_length != sequence_length:
        point, _ = _compress_and_record_cache(
            past_key_values,
            compression_config,
            quantization_config,
        )
        points.append(point)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "generation_memory.jsonl"
    _write_trace(trace_path, points)
    plot_path = write_generation_memory_plot(points, out_dir)
    return GenerationMemoryResult(
        points=tuple(points),
        generated_tokens=max(0, sequence_length - prompt_len),
        trace_path=trace_path,
        plot_path=plot_path,
    )


def write_generation_memory_plot(
    points: tuple[GenerationMemoryPoint, ...] | list[GenerationMemoryPoint],
    output_dir: str | Path,
) -> Path | None:
    """Write a dense-vs-compressed memory plot for generation points."""

    if not points:
        return None
    _configure_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    x = [point.sequence_length for point in points]
    dense = [point.dense_kv_bytes for point in points]
    compressed = [point.compressed_kv_logical_bytes for point in points]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, dense, marker="o", linewidth=2.0, label="without compression", color="#111111")
    ax.plot(x, compressed, marker="o", linewidth=2.0, label="with compression", color="#1f77b4")
    ax.set_title("KV Cache Memory During Generation")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("memory usage (bytes)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_dir = Path(output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "generation_memory.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _compress_and_record_cache(
    past_key_values: Any,
    compression_config: CompressionConfig,
    quantization_config: QuantizationConfig,
) -> tuple[GenerationMemoryPoint, Any]:
    snapshot = KVCacheSnapshot.from_past_key_values(past_key_values)
    compressed = compress_kv_snapshot_hybrid(snapshot, compression_config, quantization_config)
    memory = compressed.memory_report(snapshot)
    point = GenerationMemoryPoint(
        sequence_length=snapshot.layers[0].seq_len,
        dense_kv_bytes=memory.original_bytes,
        compressed_kv_logical_bytes=memory.total_logical_bytes,
        compressed_kv_stored_bytes=memory.total_stored_bytes,
        compression_ratio=memory.compression_ratio,
        memory_savings=1.0 - (memory.total_logical_bytes / memory.original_bytes),
    )
    reconstructed_snapshot = _cast_snapshot_like(compressed.reconstruct(), snapshot)
    reconstructed = reconstructed_snapshot.to_past_key_values_like(past_key_values)
    return point, reconstructed


def _cast_snapshot_like(reconstructed: KVCacheSnapshot, reference: KVCacheSnapshot) -> KVCacheSnapshot:
    layers: list[KVLayerCache] = []
    for reconstructed_layer, reference_layer in zip(reconstructed.layers, reference.layers, strict=True):
        layers.append(
            KVLayerCache(
                layer_idx=reconstructed_layer.layer_idx,
                key=reconstructed_layer.key.to(device=reference_layer.key.device, dtype=reference_layer.key.dtype),
                value=reconstructed_layer.value.to(
                    device=reference_layer.value.device,
                    dtype=reference_layer.value.dtype,
                ),
            )
        )
    return KVCacheSnapshot(layers=tuple(layers), metadata=reconstructed.metadata)


def _tokenize_prompt(tokenizer: Any, prompt: str, max_seq_len: int) -> dict[str, torch.Tensor]:
    try:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    except TypeError:
        encoded = tokenizer(prompt, return_tensors="pt")
    if hasattr(encoded, "data"):
        return dict(encoded.data)
    return dict(encoded)


def _attention_mask(encoded: dict[str, torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        return attention_mask
    return torch.ones_like(input_ids)


def _require_past_key_values(outputs: Any) -> Any:
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Model output did not include past_key_values; generation memory eval requires use_cache")
    return past_key_values


def _greedy_next_token(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model output did not include logits; cannot choose next generated token")
    return torch.argmax(logits[:, -1:, :], dim=-1)


def _write_trace(path: Path, points: list[GenerationMemoryPoint]) -> None:
    with path.open("w") as handle:
        for point in points:
            handle.write(json.dumps(asdict(point), sort_keys=True) + "\n")


def _configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig-")
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdgcache-")
