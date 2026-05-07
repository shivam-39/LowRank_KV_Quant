"""Configurable HuggingFace causal language model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.config import ModelConfig


@dataclass(frozen=True)
class LoadedModel:
    """Container for a HuggingFace model and tokenizer pair."""

    model: Any
    tokenizer: Any


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a config dtype string to a torch dtype."""

    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of {sorted(mapping)}")
    return mapping[normalized]


def load_causal_lm_and_tokenizer(config: ModelConfig) -> LoadedModel:
    """Load a HuggingFace causal LM and tokenizer from config.

    The import is local so core unit tests do not require importing the full
    Transformers stack unless model loading is explicitly exercised.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_torch_dtype(config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    target_device: str | None = _target_device(config.device)
    if config.device == "auto" and _accelerate_device_map_available():
        model_kwargs["device_map"] = "auto"
        target_device = None

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    except ValueError as exc:
        if model_kwargs.get("device_map") != "auto" or not _is_accelerate_device_map_error(exc):
            raise
        model_kwargs.pop("device_map")
        target_device = _target_device("auto")
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if target_device is not None:
        model = model.to(target_device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)


def _target_device(config_device: str) -> str | None:
    if config_device != "auto":
        return config_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _accelerate_device_map_available() -> bool:
    try:
        from transformers.utils import is_accelerate_available
    except Exception:
        return False
    return bool(is_accelerate_available())


def _is_accelerate_device_map_error(error: ValueError) -> bool:
    message = str(error)
    return "requires `accelerate`" in message and "device_map" in message
