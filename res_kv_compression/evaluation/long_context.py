"""Long-context windowing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LongContextWindow:
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class LongContextSummary:
    num_tokens: int
    max_seq_len: int
    stride: int
    num_windows: int
    max_window_length: int


def make_token_windows(token_ids: torch.Tensor, max_seq_len: int, stride: int) -> tuple[LongContextWindow, ...]:
    """Create overlapping token windows for long-context evaluation."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > max_seq_len:
        raise ValueError("stride must be <= max_seq_len")

    num_tokens = int(token_ids.numel())
    if num_tokens == 0:
        return ()
    windows: list[LongContextWindow] = []
    start = 0
    while start < num_tokens:
        end = min(start + max_seq_len, num_tokens)
        windows.append(LongContextWindow(start=start, end=end, length=end - start))
        if end == num_tokens:
            break
        start += stride
    return tuple(windows)


def summarize_long_context(token_ids: torch.Tensor, max_seq_len: int, stride: int) -> LongContextSummary:
    windows = make_token_windows(token_ids, max_seq_len=max_seq_len, stride=stride)
    max_window_length = max((window.length for window in windows), default=0)
    return LongContextSummary(
        num_tokens=int(token_ids.numel()),
        max_seq_len=max_seq_len,
        stride=stride,
        num_windows=len(windows),
        max_window_length=max_window_length,
    )
