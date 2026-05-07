import json

from utils.logging import ExperimentLogger


def test_experiment_logger_writes_jsonl(tmp_path) -> None:
    logger = ExperimentLogger(tmp_path)
    logger.log_config({"experiment": {"name": "unit"}})
    logger.log_metrics({"loss": 1.25}, step=3)

    config_payload = json.loads((tmp_path / "config.json").read_text())
    metrics_payload = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()

    assert config_payload["experiment"]["name"] == "unit"
    assert json.loads(metrics_payload[0])["loss"] == 1.25
    assert json.loads(metrics_payload[0])["step"] == 3
