"""Repeated offline study with task CSV and aggregate Markdown report."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.config import ExperimentConfig
from foresight_agent.prediction.risk_score import RiskScorer

from .metrics import prediction_metrics
from .offline_dataset import offline_benchmarks
from .runner import _controllers


def run_study(config: ExperimentConfig, output_dir: str | Path) -> list[dict[str, object]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    records_by_policy: dict[str, list] = defaultdict(list)

    for repeat in range(config.runs_per_task):
        run_seed = config.seed + repeat
        tasks = offline_benchmarks()
        random.Random(run_seed).shuffle(tasks)
        for policy_name in ("no_recovery", "rule_based", "oracle"):
            for case in tasks:
                # Fresh controller and planner prevent state leakage between runs.
                controller = _controllers(config)[policy_name]
                result = AgentExecutor(case.build_planner(), scorer=RiskScorer(config.risk_weights),
                                       controller=controller, max_iterations=config.max_iterations).run(case.name)
                records_by_policy[policy_name].extend(result.state.trajectory)
                rows.append({
                    "run": repeat + 1,
                    "seed": run_seed,
                    "policy": policy_name,
                    "task": case.name,
                    "failure_type": case.failure_type,
                    "task_success": case.success_check(result.state),
                    "executor_status": result.status,
                    "steps": len(result.state.trajectory),
                    "interventions": sum(record.recovery_action != "CONTINUE"
                                         for record in result.state.trajectory),
                })

    csv_path = destination / "study_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = _markdown_report(rows, records_by_policy, config)
    (destination / "study_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Run data: {csv_path}")
    return rows


def _markdown_report(rows: list[dict[str, object]], records_by_policy: dict[str, list],
                     config: ExperimentConfig) -> str:
    lines = ["# ForesightAgent Offline Study", "",
             f"Tasks per policy/run: 24  ", f"Repeated runs: {config.runs_per_task}  ",
             f"Base seed: {config.seed}", "", "## Overall results", "",
             "| Policy | Success rate | Avg. steps | Avg. interventions | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    policies = ("no_recovery", "rule_based", "oracle")
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        rate = sum(bool(row["task_success"]) for row in selected) / len(selected)
        steps = sum(int(row["steps"]) for row in selected) / len(selected)
        interventions = sum(int(row["interventions"]) for row in selected) / len(selected)
        metrics = prediction_metrics(records_by_policy[policy], config.prediction_threshold)
        lines.append(f"| {policy} | {rate:.1%} | {steps:.2f} | {interventions:.2f} | "
                     f"{metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} |")

    lines.extend(["", "## Success by failure type", "",
                  "| Failure type | No recovery | Rule-based | Oracle |", "|---|---:|---:|---:|"])
    failure_types = sorted({str(row["failure_type"]) for row in rows})
    for failure_type in failure_types:
        values = []
        for policy in policies:
            selected = [row for row in rows if row["policy"] == policy and row["failure_type"] == failure_type]
            values.append(sum(bool(row["task_success"]) for row in selected) / len(selected))
        lines.append(f"| {failure_type} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} |")
    lines.extend(["", "This synthetic study validates experimental plumbing and policy behavior. "
                  "It does not establish real-world generalization."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/v04"))
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    run_study(config, args.output_dir)


if __name__ == "__main__":
    main()
