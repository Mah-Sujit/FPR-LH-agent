"""Train and evaluate predictors on validated, manually labeled trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from foresight_agent import __version__
from foresight_agent.config import ExperimentConfig
from foresight_agent.data.external import DatasetValidationError, load_trajectories
from foresight_agent.data.quality import quality_report
from foresight_agent.prediction.features import FEATURE_NAMES, feature_vector

from .training import (_best_threshold, _classification_metrics, _models, _select, _split)


def train_external(input_path: str | Path, output_dir: str | Path,
                   config: ExperimentConfig | None = None) -> dict[str, Path]:
    import joblib

    config = config or ExperimentConfig()
    items = load_trajectories(input_path, require_labels=True)
    task_groups = [f"{item.dataset_id}:{item.task_id}" for item in items]
    if len(set(task_groups)) < 15:
        raise DatasetValidationError("External training requires at least 15 distinct task groups")
    records = [item.to_record() for item in items]
    features = [feature_vector(record) for record in records]
    labels = [int(bool(item.actual_failure)) for item in items]
    if min(labels.count(0), labels.count(1)) < 6:
        raise DatasetValidationError("External training requires at least 6 rows from each class")
    train_idx, validation_idx, test_idx = _split(features, labels, task_groups, config.seed)

    results, models = [], {}
    for name, model in _models(config.seed).items():
        model.fit(_select(features, train_idx), _select(labels, train_idx))
        validation_probabilities = model.predict_proba(_select(features, validation_idx))[:, 1]
        threshold = _best_threshold(_select(labels, validation_idx), validation_probabilities)
        test_probabilities = model.predict_proba(_select(features, test_idx))[:, 1]
        results.append({"model": name, "threshold": threshold,
                        **_classification_metrics(_select(labels, test_idx), test_probabilities, threshold)})
        models[name] = model
    best = max(results, key=lambda row: row["f1"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"model": output / "external_model.joblib", "metrics": output / "external_metrics.csv",
             "metadata": output / "external_metadata.json", "quality": output / "data_quality.json",
             "splits": output / "external_splits.json", "report": output / "external_report.md"}
    joblib.dump(models[str(best["model"])], paths["model"])
    _write_csv(paths["metrics"], results)
    paths["quality"].write_text(json.dumps(quality_report(items), indent=2), encoding="utf-8")
    manifest = {"seed": config.seed, "group_key": "dataset_id:task_id",
                "train_groups": sorted(set(_select(task_groups, train_idx))),
                "validation_groups": sorted(set(_select(task_groups, validation_idx))),
                "test_groups": sorted(set(_select(task_groups, test_idx)))}
    paths["splits"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {"version": __version__, "source": str(input_path), "selected_model": best["model"],
                "threshold": best["threshold"], "features": FEATURE_NAMES,
                "rows": len(items), "task_groups": len(set(task_groups)), "seed": config.seed}
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = _report(results, metadata)
    paths["report"].write_text(report, encoding="utf-8")
    print(report)
    return paths


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(results: list[dict], metadata: dict) -> str:
    lines = ["# External Trajectory Evaluation", "",
             f"Rows: {metadata['rows']}  ", f"Task groups: {metadata['task_groups']}  ",
             f"Selected model: **{metadata['selected_model']}**  ",
             f"Validation-selected threshold: **{metadata['threshold']:.2f}**", "",
             "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                     f"{row['f1']:.3f} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | "
                     f"{row['brier']:.3f} | {row['calibration_error']:.3f} |")
    lines.extend(["", "Metrics are valid only to the extent that task sampling and human labels are reliable."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/v07"))
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    train_external(args.input, args.output_dir, config)


if __name__ == "__main__":
    try:
        main()
    except (DatasetValidationError, ValueError) as exc:
        raise SystemExit(f"External training cannot start: {exc}") from None
