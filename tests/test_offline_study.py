from foresight_agent.config import ExperimentConfig
from foresight_agent.evaluation.offline_dataset import offline_benchmarks
from foresight_agent.evaluation.study import run_study


def test_offline_dataset_has_24_labeled_tasks():
    tasks = offline_benchmarks()
    assert len(tasks) == 24
    assert len({task.name for task in tasks}) == 24
    assert {task.failure_type for task in tasks} == {
        "none", "tool_execution_error", "repetition_loop", "missing_evidence",
        "planning_error", "tool_selection_error", "contradiction", "hallucination",
    }


def test_repeated_study_is_reproducible_and_writes_reports(tmp_path):
    config = ExperimentConfig(runs_per_task=2, seed=7)
    first = run_study(config, tmp_path / "first")
    second = run_study(config, tmp_path / "second")
    assert first == second
    assert len(first) == 24 * 3 * 2
    assert (tmp_path / "first" / "study_runs.csv").exists()
    report = (tmp_path / "first" / "study_report.md").read_text(encoding="utf-8")
    assert "Success by failure type" in report
    assert "rule_based" in report

    rates = {}
    for policy in ("no_recovery", "rule_based", "oracle"):
        selected = [row for row in first if row["policy"] == policy]
        rates[policy] = sum(bool(row["task_success"]) for row in selected) / len(selected)
    assert rates["rule_based"] > rates["no_recovery"]
    assert rates["oracle"] >= rates["rule_based"]
