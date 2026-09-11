"""Version 0.5 noisy study, threshold sweep, calibration, and cost report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.config import ExperimentConfig
from foresight_agent.prediction.risk_score import RiskScorer

from .metrics import prediction_metrics
from .noisy_dataset import noisy_benchmarks
from .policies import NoRecoveryController, OracleRecoveryController
from .recovery_factory import controller_options
from .statistics import calibration_rows, wilson_interval


def _controller(policy: str, config: ExperimentConfig, threshold: float):
    options = controller_options(config, threshold)
    if policy == "no_recovery":
        return NoRecoveryController(**options)
    if policy == "oracle":
        return OracleRecoveryController(**options)
    from foresight_agent.recovery.controller import RecoveryController
    return RecoveryController(**options)


def _execute(config: ExperimentConfig, policy: str, threshold: float) -> tuple[list[dict[str, object]], list]:
    rows, records = [], []
    for run in range(config.noisy_runs_per_task):
        seed = config.seed + run * 100_003
        for case in noisy_benchmarks():
            result = AgentExecutor(case.build_planner(seed), scorer=RiskScorer(config.risk_weights),
                                   controller=_controller(policy, config, threshold),
                                   max_iterations=config.max_iterations).run(case.name)
            records.extend(result.state.trajectory)
            interventions = sum(record.recovery_action != "CONTINUE" for record in result.state.trajectory)
            steps = len(result.state.trajectory)
            rows.append({"run": run + 1, "seed": seed, "policy": policy, "threshold": threshold,
                         "task": case.name, "failure_type": case.failure_type,
                         "task_success": case.success_check(result.state), "status": result.status,
                         "steps": steps, "interventions": interventions,
                         "simulated_latency_ms": steps * 100 + interventions * 50,
                         "simulated_cost_units": round(steps + interventions * 0.5, 2)})
    return rows, records


def run_noisy_study(config: ExperimentConfig, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main_rows, records_by_policy = [], {}
    for policy in ("no_recovery", "rule_based", "oracle"):
        rows, records = _execute(config, policy, config.prediction_threshold)
        main_rows.extend(rows)
        records_by_policy[policy] = records

    sweep_rows = []
    for threshold in config.threshold_sweep:
        rows, records = _execute(config, "rule_based", threshold)
        metrics = prediction_metrics(records, threshold)
        successes = sum(bool(row["task_success"]) for row in rows)
        tp = sum(record.risk_score >= threshold and record.actual_failure for record in records)
        fp = sum(record.risk_score >= threshold and not record.actual_failure for record in records)
        fn = sum(record.risk_score < threshold and record.actual_failure for record in records)
        tn = sum(record.risk_score < threshold and not record.actual_failure for record in records)
        sweep_rows.append({"threshold": threshold, "task_success_rate": successes / len(rows),
                           "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1,
                           "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
                           "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                           "true_positive": tp, "false_positive": fp, "false_negative": fn,
                           "true_negative": tn})

    all_records = [record for records in records_by_policy.values() for record in records]
    calibration = calibration_rows([(record.risk_score, record.actual_failure) for record in all_records])
    paths = {"runs": output / "noisy_runs.csv", "sweep": output / "threshold_sweep.csv",
             "calibration": output / "calibration.csv", "report": output / "noisy_report.md"}
    _write_csv(paths["runs"], main_rows)
    _write_csv(paths["sweep"], sweep_rows)
    _write_csv(paths["calibration"], calibration)
    report = _report(main_rows, records_by_policy, sweep_rows, config)
    paths["report"].write_text(report, encoding="utf-8")
    print(report)
    return paths


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict[str, object]], records_by_policy: dict[str, list],
            sweep: list[dict[str, object]], config: ExperimentConfig) -> str:
    lines = ["# ForesightAgent Noisy Study", "", f"Tasks: 30 × {config.noisy_runs_per_task} runs per policy  ",
             f"Base seed: {config.seed}", "", "## Policy comparison", "",
             "| Policy | Task success (95% CI) | Avg steps | Avg interventions | Avg cost | Precision | Recall | F1 | Unnecessary interventions |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in ("no_recovery", "rule_based", "oracle"):
        selected = [row for row in rows if row["policy"] == policy]
        successes = sum(bool(row["task_success"]) for row in selected)
        low, high = wilson_interval(successes, len(selected))
        metrics = prediction_metrics(records_by_policy[policy], config.prediction_threshold)
        average = lambda key: sum(float(row[key]) for row in selected) / len(selected)
        lines.append(f"| {policy} | {successes / len(selected):.1%} ({low:.1%}–{high:.1%}) | "
                     f"{average('steps'):.2f} | {average('interventions'):.2f} | "
                     f"{average('simulated_cost_units'):.2f} | {metrics.precision:.3f} | "
                     f"{metrics.recall:.3f} | {metrics.f1:.3f} | "
                     f"{metrics.unnecessary_intervention_rate:.1%} |")
    best = max(sweep, key=lambda row: float(row["f1"]))
    lines.extend(["", "## Threshold sweep", "",
                  f"Best observed F1 threshold: **{float(best['threshold']):.2f}** "
                  f"(F1 {float(best['f1']):.3f}).", "",
                  "Raw confusion matrices, calibration bins, and threshold results are included as CSV files.", "",
                  "Simulated latency and cost are comparative units, not provider billing estimates. "
                  "Results measure seeded synthetic noise and do not establish external validity."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/v05"))
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    run_noisy_study(config, args.output_dir)


if __name__ == "__main__":
    main()
