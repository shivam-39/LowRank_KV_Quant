"""KV cache extraction and serialization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class KVLayerCache:
    """Per-layer K/V tensors normalized to ``[batch, heads, tokens, head_dim]``."""

    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError(
                f"Layer {self.layer_idx} key/value shapes must match, got "
                f"{tuple(self.key.shape)} and {tuple(self.value.shape)}"
            )
        if self.key.ndim != 4:
            raise ValueError(
                f"Layer {self.layer_idx} tensors must have shape [B, H, T, D], "
                f"got {tuple(self.key.shape)}"
            )

    @property
    def num_heads(self) -> int:
        return int(self.key.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.key.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.key.shape[3])

    @property
    def bytes(self) -> int:
        return _tensor_nbytes(self.key) + _tensor_nbytes(self.value)


@dataclass(frozen=True)
class KVCacheSnapshot:
    """Captured KV cache for all transformer layers."""

    layers: tuple[KVLayerCache, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("KVCacheSnapshot requires at least one layer")

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def total_tokens(self) -> int:
        return sum(layer.key.shape[0] * layer.seq_len for layer in self.layers)

    @property
    def total_bytes(self) -> int:
        return sum(layer.bytes for layer in self.layers)

    def manifest(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "num_layers": self.num_layers,
            "total_bytes": self.total_bytes,
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "key_shape": list(layer.key.shape),
                    "value_shape": list(layer.value.shape),
                    "dtype": str(layer.key.dtype),
                    "bytes": layer.bytes,
                }
                for layer in self.layers
            ],
        }

    def save(self, output_dir: str | Path) -> None:
        """Save per-layer tensors plus a JSON manifest."""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            torch.save(
                {"key": layer.key.detach().cpu(), "value": layer.value.detach().cpu()},
                path / f"layer_{layer.layer_idx:03d}.pt",
            )
        (path / "manifest.json").write_text(json.dumps(self.manifest(), indent=2, sort_keys=True))

    @classmethod
    def from_past_key_values(
        cls,
        past_key_values: Any,
        metadata: dict[str, Any] | None = None,
    ) -> "KVCacheSnapshot":
        """Build a snapshot from HuggingFace ``past_key_values`` output."""

        legacy_cache = _to_legacy_cache(past_key_values)
        layers: list[KVLayerCache] = []
        for layer_idx, layer_cache in enumerate(legacy_cache):
            key, value = _extract_key_value(layer_cache)
            layers.append(
                KVLayerCache(
                    layer_idx=layer_idx,
                    key=_normalize_kv_tensor(key),
                    value=_normalize_kv_tensor(value),
                )
            )
        return cls(layers=tuple(layers), metadata=metadata or {})

    def to_legacy_past_key_values(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert the snapshot to HuggingFace's legacy tuple cache format."""

        return tuple((layer.key, layer.value) for layer in self.layers)

    def to_past_key_values_like(self, reference: Any | None = None) -> Any:
        """Convert the snapshot to a cache object compatible with ``reference`` when possible."""

        legacy_cache = self.to_legacy_past_key_values()
        if reference is not None and _is_cache_object(reference):
            try:
                return type(reference).from_legacy_cache(legacy_cache)
            except Exception:
                pass
            try:
                return type(reference)(legacy_cache)
            except Exception:
                pass
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache(legacy_cache)
            except Exception:
                return legacy_cache
        return legacy_cache


class KVCacheExtractor:
    """Extract KV caches from causal LM forward passes."""

    @torch.no_grad()
    def extract_from_inputs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KVCacheSnapshot:
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
            raise RuntimeError("Model output did not include past_key_values; ensure use_cache=True is supported")

        snapshot_metadata = {
            "source": "past_key_values",
            "input_shape": list(input_ids.shape),
        }
        if metadata:
            snapshot_metadata.update(metadata)
        return KVCacheSnapshot.from_past_key_values(outputs.past_key_values, metadata=snapshot_metadata)

    @torch.no_grad()
    def extract_from_text(
        self,
        loaded_model: Any,
        text: str,
        max_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KVCacheSnapshot:
        tokenizer = loaded_model.tokenizer
        model = loaded_model.model
        encoded = tokenizer(text, return_tensors="pt", truncation=max_length is not None, max_length=max_length)
        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return self.extract_from_inputs(model, input_ids, attention_mask=attention_mask, metadata=metadata)


class AttentionHookCapture:
    """Forward-hook collector for attention modules that return present K/V."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured: list[KVLayerCache] = []

    def register(self, model: torch.nn.Module, name_filter: str = "attn") -> None:
        """Register hooks on modules whose qualified name contains ``name_filter``."""

        lowered_filter = name_filter.lower()
        for module_name, module in model.named_modules():
            class_name = module.__class__.__name__.lower()
            if lowered_filter in module_name.lower() or "attention" in class_name:
                handle = module.register_forward_hook(self._make_hook(module_name))
                self._handles.append(handle)

    def clear(self) -> None:
        self._captured.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def snapshot(self, metadata: dict[str, Any] | None = None) -> KVCacheSnapshot:
        if not self._captured:
            raise RuntimeError("No KV tensors were captured by attention hooks")
        layers = tuple(
            KVLayerCache(layer_idx=idx, key=layer.key, value=layer.value)
            for idx, layer in enumerate(self._captured)
        )
        return KVCacheSnapshot(layers=layers, metadata=metadata or {"source": "attention_hooks"})

    def _make_hook(self, module_name: str) -> Any:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            present = _find_present_key_value(output)
            if present is None:
                return
            key, value = present
            self._captured.append(
                KVLayerCache(
                    layer_idx=len(self._captured),
                    key=_normalize_kv_tensor(key),
                    value=_normalize_kv_tensor(value),
                )
            )

        hook.__name__ = f"capture_kv_{module_name.replace('.', '_')}"
        return hook


def _to_legacy_cache(past_key_values: Any) -> Iterable[Any]:
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _is_cache_object(past_key_values: Any) -> bool:
    return hasattr(past_key_values, "get_seq_length") or hasattr(past_key_values, "to_legacy_cache")


def _extract_key_value(layer_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer_cache, dict):
        return layer_cache["key"], layer_cache["value"]
    if isinstance(layer_cache, SimpleNamespace):
        return layer_cache.key, layer_cache.value
    if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
        return layer_cache[0], layer_cache[1]
    if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
        return layer_cache.keys, layer_cache.values
    raise TypeError(f"Unsupported layer cache type: {type(layer_cache)!r}")


def _normalize_kv_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"KV cache entries must be tensors, got {type(tensor)!r}")
    if tensor.ndim != 4:
        raise ValueError(f"KV cache tensor must be rank 4, got shape {tuple(tensor.shape)}")
    return tensor.detach()


def _find_present_key_value(output: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(output, dict):
        for key in ("past_key_value", "present_key_value", "present"):
            if key in output:
                return _extract_key_value(output[key])
        return None
    if isinstance(output, SimpleNamespace):
        for attr in ("past_key_value", "present_key_value", "present"):
            if hasattr(output, attr):
                return _extract_key_value(getattr(output, attr))
        return None
    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if _looks_like_key_value(item):
                return _extract_key_value(item)
    return None


def _looks_like_key_value(item: Any) -> bool:
    if isinstance(item, dict):
        return "key" in item and "value" in item
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return isinstance(item[0], torch.Tensor) and isinstance(item[1], torch.Tensor)
    return hasattr(item, "key") and hasattr(item, "value")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
