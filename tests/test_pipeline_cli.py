import sys

import pytest

from foresight_agent.research import pipeline


def test_pipeline_cli_reports_missing_dataset_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", "--dataset", "missing-dataset.jsonl"])
    with pytest.raises(FileNotFoundError):
        pipeline.main()
