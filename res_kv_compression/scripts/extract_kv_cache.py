"""Extract and save a HuggingFace model KV cache for a prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kv_cache import KVCacheExtractor
from models.loader import load_causal_lm_and_tokenizer
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompt", default="Low-rank KV cache compression studies")
    parser.add_argument("--output-dir", default="logs/kv_cache")
    args = parser.parse_args()

    config = load_config(args.config)
    loaded_model = load_causal_lm_and_tokenizer(config.model)
    snapshot = KVCacheExtractor().extract_from_text(
        loaded_model,
        args.prompt,
        max_length=config.model.max_seq_len,
        metadata={"model_name": config.model.model_name},
    )
    snapshot.save(args.output_dir)
    print(f"saved {snapshot.num_layers} KV layers to {args.output_dir}")


if __name__ == "__main__":
    main()
