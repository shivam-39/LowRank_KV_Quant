"""Perplexity evaluation for causal language models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class PerplexityResult:
    nll: float
    perplexity: float
    num_tokens: int
    num_windows: int


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_seq_len: int,
    stride: int,
    device: torch.device | str | None = None,
) -> PerplexityResult:
    """Evaluate sliding-window perplexity over provided texts."""

    if not texts:
        raise ValueError("texts must be non-empty")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    joined = "\n\n".join(text for text in texts if text)
    encoded = tokenizer(joined, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    num_windows = 0
    previous_end = 0
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_seq_len, input_ids.size(1))
        target_len = end - previous_end
        if target_len <= 0:
            continue
        input_window = input_ids[:, begin:end]
        labels = input_window.clone()
        labels[:, :-target_len] = -100
        outputs = model(input_window, labels=labels)
        loss = outputs.loss
        total_nll += float(loss.item()) * target_len
        total_tokens += target_len
        num_windows += 1
        previous_end = end
        if end == input_ids.size(1):
            break

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated")
    mean_nll = total_nll / total_tokens
    return PerplexityResult(
        nll=mean_nll,
        perplexity=float(math.exp(mean_nll)),
        num_tokens=total_tokens,
        num_windows=num_windows,
    )


def load_wikitext_texts(
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_samples: int,
) -> list[str]:
    """Load WikiText-style text samples via HuggingFace datasets."""

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [row["text"] for row in dataset if row.get("text")]
    if max_samples > 0:
        texts = texts[:max_samples]
    return texts
