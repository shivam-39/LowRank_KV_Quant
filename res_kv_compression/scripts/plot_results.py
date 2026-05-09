"""Generate comparison plots from experiment logs.

Reads `config.json` + `metrics.jsonl` under one or more run directories
(typically `logs/<run_name>/`) and emits PNG plots under `plots/`.

This is intentionally lightweight: it does not require Weights & Biases or
TensorBoard and operates purely on the local JSON logs produced by
`utils.logging.ExperimentLogger`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _configure_matplotlib() -> None:
    # In restricted environments the default cache dir may be unwritable.
    # Point matplotlib to a per-process temp directory.
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig-")
    # Fontconfig caches are commonly stored under ~/.cache which may be
    # unwritable in sandboxed environments.
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdgcache-")


@dataclass(frozen=True)
class RunSeries:
    run_dir: Path
    config: dict[str, Any]
    records: tuple[dict[str, float], ...]

    @property
    def name(self) -> str:
        return self.run_dir.name


def load_run_series(run_dir: str | Path) -> RunSeries:
    path = Path(run_dir)
    config_path = path / "config.json"
    metrics_path = path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl under {path}")
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    records: list[dict[str, float]] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        # Metrics are expected to be scalar floats, but older runs may include ints.
        sanitized: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                sanitized[key] = float(value)
        if sanitized:
            records.append(sanitized)
    return RunSeries(run_dir=path, config=config, records=tuple(records))


def discover_runs(root: str | Path = "logs") -> tuple[Path, ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()
    candidates = [p for p in root_path.iterdir() if p.is_dir()]
    runs = [p for p in candidates if (p / "metrics.jsonl").exists()]
    return tuple(sorted(runs))


def _plot_series(
    ax: Any,
    x: list[int],
    y: list[float],
    label: str,
    *,
    color: str | None = None,
) -> None:
    if not y:
        return
    ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)


def _plot_constant(ax: Any, x: list[int], value: float, label: str, *, color: str = "black") -> None:
    if not x:
        return
    ax.plot([x[0], x[-1]], [value, value], linestyle="--", linewidth=1.5, label=label, color=color)


def _extract(series: RunSeries, key: str) -> list[float]:
    values: list[float] = []
    for record in series.records:
        if key in record:
            values.append(float(record[key]))
    return values


def _step_axis(series: RunSeries, key: str) -> list[int]:
    steps: list[int] = []
    for idx, record in enumerate(series.records):
        if key in record:
            steps.append(idx)
    return steps


def write_plots(series: RunSeries, output_dir: Path | None = None) -> tuple[Path, ...]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    out_dir = output_dir or (series.run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Memory: compare baseline (dense prompt KV bytes) vs compressed logical bytes.
    generation_x = _extract(series, "generation_seq_len")
    generation_dense = _extract(series, "generation_dense_kv_bytes")
    generation_compressed = _extract(series, "generation_compressed_kv_logical_bytes")
    if generation_x and generation_dense and generation_compressed:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            generation_x,
            generation_dense,
            marker="o",
            linewidth=2.0,
            label="without compression",
            color="#111111",
        )
        ax.plot(
            generation_x,
            generation_compressed,
            marker="o",
            linewidth=2.0,
            label="with compression",
            color="#1f77b4",
        )
        ax.set_title(f"Generation KV Memory ({series.name})")
        ax.set_xlabel("sequence length (tokens)")
        ax.set_ylabel("memory usage (bytes)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "generation_memory.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Memory: compare baseline (dense prompt KV bytes) vs compressed logical bytes.
    mem_steps = _step_axis(series, "prompt_kv_bytes")
    prompt_bytes = _extract(series, "prompt_kv_bytes")
    compressed_logical = _extract(series, "compressed_kv_logical_bytes")
    estimated_dense = _extract(series, "estimated_dense_kv_bytes")
    if prompt_bytes:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        # _plot_series(ax, mem_steps, prompt_bytes, "baseline: prompt_kv_bytes", color="#111111")
        if compressed_logical:
            compressed_steps = _step_axis(series, "compressed_kv_logical_bytes")
            _plot_series(ax, compressed_steps, compressed_logical, "compressed: bytes", color="#1f77b4")
        if estimated_dense:
            dense_steps = _step_axis(series, "estimated_dense_kv_bytes")
            _plot_series(ax, dense_steps, estimated_dense, "reference: dense_kv_bytes", color="#ff7f0e")
        ax.set_title(f"KV Memory Comparison ({series.name})")
        ax.set_xlabel("rank (context len = 2048)")
        ax.set_ylabel("bytes")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "memory_comparison.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Reconstruction error: compare to a zero baseline.
    recon_steps = _step_axis(series, "prompt_reconstruction_error")
    recon = _extract(series, "prompt_reconstruction_error")
    if recon:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_series(ax, recon_steps, recon, "compressed: prompt_reconstruction_error", color="#d62728")
        _plot_constant(ax, recon_steps, 0.0, "baseline: 0.0", color="#111111")
        ax.set_title(f"Relative Reconstruction Error ({series.name})")
        ax.set_xlabel("metrics.jsonl record index")
        ax.set_ylabel("relative error (Frobenius)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "reconstruction_error.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Perplexity.
    ppl_steps = _step_axis(series, "perplexity")
    perplexity = _extract(series, "perplexity")
    if perplexity:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_series(ax, ppl_steps, perplexity, "perplexity", color="#2ca02c")
        ax.set_title(f"Perplexity ({series.name})")
        ax.set_xlabel("metrics.jsonl record index")
        ax.set_ylabel("perplexity")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "perplexity.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Latency.
    latency_steps = _step_axis(series, "latency_mean_ms")
    latency_mean = _extract(series, "latency_mean_ms")
    latency_p50 = _extract(series, "latency_p50_ms")
    latency_p95 = _extract(series, "latency_p95_ms")
    if latency_mean:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_series(ax, latency_steps, latency_mean, "mean_ms", color="#1f77b4")
        if latency_p50:
            _plot_series(ax, _step_axis(series, "latency_p50_ms"), latency_p50, "p50_ms", color="#ff7f0e")
        if latency_p95:
            _plot_series(ax, _step_axis(series, "latency_p95_ms"), latency_p95, "p95_ms", color="#d62728")
        ax.set_title(f"Latency ({series.name})")
        ax.set_xlabel("metrics.jsonl record index")
        ax.set_ylabel("ms")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "latency.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Tokens / sec.
    tps_steps = _step_axis(series, "tokens_per_sec")
    tps = _extract(series, "tokens_per_sec")
    if tps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_series(ax, tps_steps, tps, "tokens_per_sec", color="#9467bd")
        ax.set_title(f"Throughput ({series.name})")
        ax.set_xlabel("metrics.jsonl record index")
        ax.set_ylabel("tokens/sec")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "throughput.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    # Memory savings / compression ratio (derived in runner when memory_eval is enabled).
    ratio_steps = _step_axis(series, "memory_compression_ratio")
    ratio = _extract(series, "memory_compression_ratio")
    savings_steps = _step_axis(series, "memory_savings")
    savings = _extract(series, "memory_savings")
    if ratio or savings:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if ratio:
            _plot_series(ax, ratio_steps, ratio, "memory_compression_ratio (orig / compressed)", color="#1f77b4")
        if savings:
            _plot_series(ax, savings_steps, savings, "memory_savings (1 - compressed/orig)", color="#ff7f0e")
            _plot_constant(ax, savings_steps, 0.0, "baseline: 0.0", color="#111111")
        ax.set_title(f"Memory Ratios ({series.name})")
        ax.set_xlabel("metrics.jsonl record index")
        ax.set_ylabel("ratio")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out_dir / "memory_ratios.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(target)

    return tuple(written)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Run directory under logs/ (repeatable). Defaults to all runs under logs/.",
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory containing run subdirectories (default: logs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dirs: Iterable[Path]
    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = discover_runs(args.logs_root)

    for run_dir in run_dirs:
        series = load_run_series(run_dir)
        written = write_plots(series)
        if written:
            print(f"[{series.name}] wrote {len(written)} plot(s) to {series.run_dir / 'plots'}")
        else:
            print(f"[{series.name}] no plottable metrics found in {series.run_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
