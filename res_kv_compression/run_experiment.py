"""Command-line entrypoint for KV compression experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from experiments.runner import run_experiment
from utils.config import apply_overrides, load_config


def _parse_cli(argv: list[str]) -> tuple[Path, list[str]]:
    config_path = Path("configs/default.yaml")
    overrides: list[str] = []
    for arg in argv:
        if arg.startswith("config="):
            config_path = Path(arg.split("=", 1)[1])
        else:
            overrides.append(arg)
    return config_path, overrides


def main(argv: list[str] | None = None) -> None:
    config_path, overrides = _parse_cli(list(sys.argv[1:] if argv is None else argv))
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)
    result = run_experiment(config)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
