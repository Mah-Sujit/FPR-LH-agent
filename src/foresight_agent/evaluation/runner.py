"""Run benchmark policies and export per-run results to CSV."""

import argparse
import csv
from pathlib import Path

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.config import ExperimentConfig
from foresight_agent.prediction.risk_score import RiskScorer
from foresight_agent.recovery.controller import RecoveryController

from .benchmarks import default_benchmarks
from .metrics import prediction_metrics
from .policies import NoRecoveryController, OracleRecoveryController
from .recovery_factory import controller_options


def _controllers(config: ExperimentConfig) -> dict[str, RecoveryController]:
    options = controller_options(config)
    return {"no_recovery": NoRecoveryController(**options), "rule_based": RecoveryController(**options),
            "oracle": OracleRecoveryController(**options)}


def run_experiment(config: ExperimentConfig, output_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_records = []
    for policy_name, controller in _controllers(config).items():
        for case in default_benchmarks():
            result = AgentExecutor(case.build_planner(), scorer=RiskScorer(config.risk_weights),
                                   controller=controller, max_iterations=config.max_iterations).run(case.name)
            all_records.extend(result.state.trajectory)
            rows.append({"policy": policy_name, "benchmark": case.name, "failure_type": case.failure_type,
                         "executor_status": result.status, "task_success": case.success_check(result.state),
                         "steps": len(result.state.trajectory),
                         "interventions": sum(r.recovery_action != "CONTINUE" for r in result.state.trajectory)})
    metrics = prediction_metrics(all_records, config.prediction_threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("ForesightAgent benchmark")
    for name in ("no_recovery", "rule_based", "oracle"):
        selected = [row for row in rows if row["policy"] == name]
        successes = sum(bool(row["task_success"]) for row in selected)
        print(f"  {name:<12} task success: {successes}/{len(selected)}")
    print(f"  prediction precision={metrics.precision:.3f} recall={metrics.recall:.3f} f1={metrics.f1:.3f}")
    print(f"  results: {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/benchmark.csv"))
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    run_experiment(config, args.output)


if __name__ == "__main__":
    main()
