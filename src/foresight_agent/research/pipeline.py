"""One-command collection, labeling, training, auditing, and research packaging."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from foresight_agent import __version__
from foresight_agent.config import ExperimentConfig
from foresight_agent.data.annotation import bootstrap_labels
from foresight_agent.data.collector import collect
from foresight_agent.data.external import DatasetValidationError
from foresight_agent.evaluation.external_training import train_external

from .artifacts import (audit_run, environment_manifest, sha256, write_checksums, write_datasheet,
                        write_metrics_svg, write_model_card)


def run_pipeline(config_path: str | Path = "experiments/config.json",
                 dataset_path: str | Path | None = None,
                 output_root: str | Path = "experiments/results/v09") -> Path:
    config_path = Path(config_path)
    config = ExperimentConfig.from_json(config_path)
    collection_manifest = {"label_mode": "human-reviewed"}
    if dataset_path is None:
        collection_dir = Path("data/collection/v09")
        collected = collect("experiments/tasks/v08_tasks.json", collection_dir)
        dataset = collection_dir / "provisional_labeled.jsonl"
        bootstrap_labels(collected["trajectories"], dataset)
        collection_manifest = json.loads(collected["manifest"].read_text(encoding="utf-8"))
        collection_manifest["label_mode"] = "automated-bootstrap-v0.8"
    else:
        dataset = Path(dataset_path)
    dataset_hash = sha256(dataset)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{dataset_hash[:8]}"
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(environment_manifest(), indent=2), encoding="utf-8")
    training = train_external(dataset, output / "model", config)
    audit = audit_run(dataset, training["splits"], training["metrics"])
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_model_card(training["metadata"], training["metrics"], dataset_hash, output / "MODEL_CARD.md")
    write_datasheet(dataset, collection_manifest, output / "DATASET_CARD.md")
    write_metrics_svg(training["metrics"], output / "metrics.svg")
    run_manifest = {"run_id": run_id, "project_version": __version__, "created_at": timestamp,
                    "dataset": str(dataset), "dataset_sha256": dataset_hash,
                    "config": str(config_path), "audit_passed": audit["passed"]}
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_checksums(output, output / "checksums.json")
    print(f"Research package: {output}")
    print(f"Audit passed: {audit['passed']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/config.json"))
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Human-reviewed JSONL/CSV; omit to use clearly marked bootstrap labels")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/v09"))
    args = parser.parse_args()
    run_pipeline(args.config, args.dataset, args.output_root)


if __name__ == "__main__":
    try:
        main()
    except (DatasetValidationError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Research pipeline stopped: {exc}") from None
