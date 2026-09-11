"""Version 0.6 group-split training and learned-recovery evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.config import ExperimentConfig
from foresight_agent.prediction.features import FEATURE_NAMES, feature_vector
from foresight_agent.prediction.learned import LearnedRiskScorer
from foresight_agent.prediction.risk_score import RiskScorer
from foresight_agent.recovery.controller import RecoveryController

from .noisy_dataset import noisy_benchmarks
from .policies import NoRecoveryController
from .recovery_factory import controller_options
from .statistics import calibration_rows


def build_training_data(config: ExperimentConfig, runs: int | None = None):
    """Collect unbiased one-step trajectories with stable task group IDs."""
    rows, labels, groups, records = [], [], [], []
    total_runs = runs or max(30, config.noisy_runs_per_task)
    for run in range(total_runs):
        seed = config.seed + run * 100_003
        for case in noisy_benchmarks():
            result = AgentExecutor(
                case.build_planner(seed),
                scorer=RiskScorer(config.risk_weights),
                controller=NoRecoveryController(),
                max_iterations=config.max_iterations,
            ).run(case.name)
            for record in result.state.trajectory:
                rows.append(feature_vector(record))
                labels.append(int(record.actual_failure))
                groups.append(case.name)
                records.append(record)
    return rows, labels, groups, records


def _models(seed: int):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    raw = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=seed)),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=8,
                                                  class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=4,
                                                  class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    return {name: CalibratedClassifierCV(model, method="sigmoid", cv=3) for name, model in raw.items()}


def _classification_metrics(labels, probabilities, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    predictions = [probability >= threshold for probability in probabilities]
    bins = calibration_rows(list(zip(probabilities, map(bool, labels))), bins=10)
    ece = sum(row["count"] * abs(row["mean_risk"] - row["observed_failure_rate"])
              for row in bins) / len(labels)
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "calibration_error": ece,
    }


def _best_threshold(labels, probabilities) -> float:
    candidates = [index / 100 for index in range(5, 96, 5)]
    return max(candidates, key=lambda value: _classification_metrics(labels, probabilities, value)["f1"])


def _split(rows, labels, groups, seed: int):
    from sklearn.model_selection import GroupShuffleSplit
    for attempt in range(200):
        train_indices, holdout_indices = next(GroupShuffleSplit(
            n_splits=1, test_size=0.4, random_state=seed + attempt).split(rows, labels, groups))
        holdout_groups = [groups[index] for index in holdout_indices]
        validation_relative, test_relative = next(GroupShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed + 1_000 + attempt).split(
                holdout_indices, [labels[index] for index in holdout_indices], holdout_groups))
        validation_indices = [holdout_indices[index] for index in validation_relative]
        test_indices = [holdout_indices[index] for index in test_relative]
        partitions = (list(train_indices), validation_indices, test_indices)
        train_labels = _select(labels, partitions[0])
        if (all(set(_select(labels, partition)) == {0, 1} for partition in partitions)
                and min(train_labels.count(0), train_labels.count(1)) >= 3):
            return partitions
    raise ValueError(
        "Could not create task-grouped train/validation/test splits containing both classes. "
        "Collect failure and non-failure examples across more distinct tasks."
    )


def _select(values, indices):
    return [values[index] for index in indices]


def train_models(config: ExperimentConfig, output_dir: str | Path, runs: int | None = None) -> dict[str, Path]:
    import joblib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, labels, groups, records = build_training_data(config, runs)
    train_idx, validation_idx, test_idx = _split(rows, labels, groups, config.seed)
    model_rows, fitted = [], {}

    rule_validation = [records[index].risk_score for index in validation_idx]
    rule_threshold = _best_threshold(_select(labels, validation_idx), rule_validation)
    rule_test = [records[index].risk_score for index in test_idx]
    rule_metrics = _classification_metrics(_select(labels, test_idx), rule_test, rule_threshold)
    model_rows.append({"model": "rule_based", "threshold": rule_threshold, **rule_metrics})

    for name, model in _models(config.seed).items():
        model.fit(_select(rows, train_idx), _select(labels, train_idx))
        validation_probabilities = model.predict_proba(_select(rows, validation_idx))[:, 1]
        threshold = _best_threshold(_select(labels, validation_idx), validation_probabilities)
        test_probabilities = model.predict_proba(_select(rows, test_idx))[:, 1]
        metrics = _classification_metrics(_select(labels, test_idx), test_probabilities, threshold)
        model_rows.append({"model": name, "threshold": threshold, **metrics})
        fitted[name] = model

    best = max((row for row in model_rows if row["model"] != "rule_based"), key=lambda row: row["f1"])
    best_model = fitted[str(best["model"])]
    paths = {"model": output / "best_model.joblib", "metadata": output / "model_metadata.json",
             "metrics": output / "model_metrics.csv", "splits": output / "split_manifest.json",
             "report": output / "training_report.md", "recovery": output / "recovery_comparison.csv"}
    joblib.dump(best_model, paths["model"])
    split_manifest = {"seed": config.seed, "strategy": "grouped_by_task",
                      "train_groups": sorted(set(_select(groups, train_idx))),
                      "validation_groups": sorted(set(_select(groups, validation_idx))),
                      "test_groups": sorted(set(_select(groups, test_idx)))}
    paths["splits"].write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    metadata = {"version": "0.6.0", "model": best["model"], "threshold": best["threshold"],
                "features": FEATURE_NAMES, "training_rows": len(train_idx),
                "validation_rows": len(validation_idx), "test_rows": len(test_idx),
                "seed": config.seed}
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_csv(paths["metrics"], model_rows)
    recovery_rows = _recovery_evaluation(config, best_model, float(best["threshold"]),
                                         set(split_manifest["test_groups"]))
    _write_csv(paths["recovery"], recovery_rows)
    report = _report(model_rows, metadata, recovery_rows)
    paths["report"].write_text(report, encoding="utf-8")
    print(report)
    return paths


def _recovery_evaluation(config, model, learned_threshold: float, test_groups: set[str]):
    cases = [case for case in noisy_benchmarks() if case.name in test_groups]
    rows = []
    policies = {
        "no_recovery": (RiskScorer(config.risk_weights), NoRecoveryController()),
        "rule_based": (RiskScorer(config.risk_weights), RecoveryController(**controller_options(config))),
        "learned": (LearnedRiskScorer(model), RecoveryController(**controller_options(config, learned_threshold))),
    }
    for policy, (scorer, controller) in policies.items():
        successes, total_steps = 0, 0
        for run in range(20):
            seed = config.seed + 900_000 + run * 100_003
            for case in cases:
                result = AgentExecutor(case.build_planner(seed), scorer=scorer, controller=controller,
                                       max_iterations=config.max_iterations).run(case.name)
                successes += case.success_check(result.state)
                total_steps += len(result.state.trajectory)
        total = len(cases) * 20
        rows.append({"policy": policy, "tasks": total, "successes": successes,
                     "task_success_rate": successes / total, "average_steps": total_steps / total})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(metrics, metadata, recovery) -> str:
    lines = ["# ForesightAgent Learned Predictor Study", "",
             f"Selected model: **{metadata['model']}**  ",
             f"Validation-selected threshold: **{metadata['threshold']:.2f}**  ",
             "Split: task-grouped train/validation/test (no task appears in multiple splits)", "",
             "## Held-out prediction results", "",
             "| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['model']} | {row['threshold']:.2f} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} | "
                     f"{row['pr_auc']:.3f} | {row['brier']:.3f} | {row['calibration_error']:.3f} |")
    lines.extend(["", "## Recovery on held-out task groups", "",
                  "| Policy | Task success | Average steps |", "|---|---:|---:|"])
    for row in recovery:
        lines.append(f"| {row['policy']} | {row['task_success_rate']:.1%} | {row['average_steps']:.2f} |")
    lines.extend(["", "Models are trained only on seeded synthetic trajectories. External validation is required."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/v06"))
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    train_models(config, args.output_dir, args.runs)


if __name__ == "__main__":
    main()
