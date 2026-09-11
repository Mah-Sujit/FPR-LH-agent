import csv

from foresight_agent.config import ExperimentConfig
from foresight_agent.evaluation.runner import run_experiment


def test_comparative_experiment_exports_csv(tmp_path):
    output = tmp_path / "results.csv"
    rows = run_experiment(ExperimentConfig(), output)
    assert len(rows) == 12
    assert output.exists()
    with output.open(encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 12

    success = {(row["policy"], row["benchmark"]): row["task_success"] for row in rows}
    assert success[("no_recovery", "transient_tool_failure")] is False
    assert success[("rule_based", "transient_tool_failure")] is True
    assert success[("rule_based", "low_confidence")] is True
    assert success[("oracle", "low_confidence")] is True

