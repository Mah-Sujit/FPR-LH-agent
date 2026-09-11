import csv

from foresight_agent.config import ExperimentConfig
from foresight_agent.evaluation.noisy_dataset import noisy_benchmarks
from foresight_agent.evaluation.noisy_study import run_noisy_study
from foresight_agent.evaluation.statistics import calibration_rows, wilson_interval


def test_noisy_dataset_has_30_seeded_cases():
    cases = noisy_benchmarks()
    assert len(cases) == 30
    assert len({case.name for case in cases}) == 30


def test_statistical_helpers_are_bounded():
    low, high = wilson_interval(7, 10)
    assert 0.0 < low < 0.7 < high < 1.0
    rows = calibration_rows([(0.1, False), (0.2, True), (0.9, True)])
    assert sum(row["count"] for row in rows) == 3
    assert all(0.0 <= row["observed_failure_rate"] <= 1.0 for row in rows)


def test_noisy_study_writes_all_artifacts(tmp_path):
    config = ExperimentConfig(noisy_runs_per_task=2, threshold_sweep=[0.2, 0.5, 0.8], seed=19)
    paths = run_noisy_study(config, tmp_path)
    assert all(path.exists() for path in paths.values())
    with paths["runs"].open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30 * 2 * 3
    with paths["sweep"].open(encoding="utf-8") as stream:
        sweep = list(csv.DictReader(stream))
    assert len(sweep) == 3
    assert {"true_positive_rate", "false_positive_rate"} <= set(sweep[0])
    assert "95% CI" in paths["report"].read_text(encoding="utf-8")
