"""Latency benchmarking helpers."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class LatencyResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    iterations: int
    tokens_per_sec: float | None = None


def benchmark_callable(
    fn: Callable[[], T],
    warmup: int = 1,
    iterations: int = 5,
    tokens_per_iteration: int | None = None,
) -> LatencyResult:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    for _ in range(warmup):
        fn()
        _synchronize()

    timings_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(timings_ms)
    tokens_per_sec = None
    if tokens_per_iteration is not None and mean_ms > 0:
        tokens_per_sec = tokens_per_iteration / (mean_ms / 1000.0)
    return LatencyResult(
        mean_ms=mean_ms,
        p50_ms=float(statistics.median(timings_ms)),
        p95_ms=_percentile(timings_ms, 0.95),
        iterations=iterations,
        tokens_per_sec=tokens_per_sec,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return float(sorted_values[index])


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
