from experiments.runner import run_experiment
from utils.config import apply_overrides, load_config


def test_runner_completes_setup_without_model_loading(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    config = apply_overrides(
        config,
        [
            f"logging.output_dir={tmp_path}",
            "evaluation.perplexity_eval=false",
            "evaluation.long_context_eval=false",
            "evaluation.latency_eval=false",
            "evaluation.memory_eval=false",
        ],
    )

    result = run_experiment(config)

    assert result.metadata["status"] == "setup_complete"
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()
