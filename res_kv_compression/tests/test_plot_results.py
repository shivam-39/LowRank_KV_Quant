from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_results import load_run_series


def test_load_run_series_reads_metrics_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"experiment": {"name": "toy"}}))
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"setup_complete": 1.0}),
                json.dumps({"prompt_kv_bytes": 128, "compressed_kv_logical_bytes": 64.0}),
                json.dumps({"perplexity": 10.0, "latency_mean_ms": 1.2}),
            ]
        )
        + "\n"
    )

    series = load_run_series(run_dir)

    assert series.name == "run"
    assert series.config["experiment"]["name"] == "toy"
    assert any("prompt_kv_bytes" in record for record in series.records)
    assert any("perplexity" in record for record in series.records)

