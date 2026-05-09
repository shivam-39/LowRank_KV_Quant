from pathlib import Path

import pytest

from utils.config import apply_overrides, config_to_dict, load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.compression.rank == 16
    assert config.quantization.quant_bits == 4
    assert config.objectives.lambda_attention == 1.0
    assert config.evaluation.generation_memory_eval is False


def test_apply_overrides_updates_nested_values() -> None:
    config = load_config(Path("configs/default.yaml"))
    updated = apply_overrides(
        config,
        [
            "compression.rank=8",
            "quantization.quant_bits=8",
            "evaluation.perplexity_eval=true",
            "evaluation.generation_memory_eval=true",
        ],
    )

    assert updated.compression.rank == 8
    assert updated.quantization.quant_bits == 8
    assert updated.evaluation.perplexity_eval is True
    assert updated.evaluation.generation_memory_eval is True


def test_config_to_dict_is_plain_mapping() -> None:
    config = load_config(Path("configs/default.yaml"))
    payload = config_to_dict(config)

    assert payload["compression"]["rank"] == 16
    assert payload["logging"]["output_dir"] == "logs/default"


def test_invalid_quant_bits_are_rejected() -> None:
    config = load_config(Path("configs/default.yaml"))

    with pytest.raises(ValueError, match="quant_bits"):
        apply_overrides(config, ["quantization.quant_bits=3"])
