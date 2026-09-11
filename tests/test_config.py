import json

import pytest

from foresight_agent.config import ExperimentConfig


def test_config_loads_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"max_retries": 4}), encoding="utf-8")
    config = ExperimentConfig.from_json(path)
    assert config.max_retries == 4
    assert config.prediction_threshold == 0.30


def test_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mystery": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        ExperimentConfig.from_json(path)

