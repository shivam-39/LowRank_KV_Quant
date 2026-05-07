"""Smoke benchmark for project setup and configuration loading."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import config_to_dict, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    start = time.perf_counter()
    config = load_config(args.config)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = {
        "benchmark": "setup",
        "elapsed_ms": elapsed_ms,
        "experiment": config.experiment.name,
        "config": config_to_dict(config),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
