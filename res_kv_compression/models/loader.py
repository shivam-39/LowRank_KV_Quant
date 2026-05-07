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
        "torch_dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if config.device not in {"auto", "cpu"}:
        model = model.to(config.device)
    elif config.device == "cpu":
        model = model.to("cpu")
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)
