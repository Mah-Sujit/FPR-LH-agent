"""Dual annotation, agreement measurement, and conflict adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .external import DatasetValidationError, ExternalTrajectory, load_trajectories, write_normalized_jsonl
from .quality import write_annotation_template


def create_assignment(input_path: str | Path, output_path: str | Path, annotator_id: str) -> None:
    items = load_trajectories(input_path)
    for item in items:
        item.actual_failure = None
        item.failure_type = None
        item.annotator_id = annotator_id
    write_annotation_template(items, output_path)


def bootstrap_labels(input_path: str | Path, output_path: str | Path) -> None:
    """Create transparent provisional labels from deterministic tool outcomes."""
    items = load_trajectories(input_path)
    for item in items:
        if not item.success:
            item.actual_failure, item.failure_type = True, "tool_execution_error"
        elif item.action.startswith("search_text") and not item.observation.strip():
            item.actual_failure, item.failure_type = True, "missing_evidence"
        else:
            item.actual_failure, item.failure_type = False, "none"
        item.annotator_id = "automated-bootstrap-v0.8"
        item.source = f"{item.source or 'unknown'}:provisional-label"
    destination = Path(output_path)
    if destination.suffix.lower() == ".csv":
        write_annotation_template(items, destination)
    else:
        write_normalized_jsonl(items, destination)
    print(f"Created {len(items)} provisional labels: {output_path}")


def _key(item: ExternalTrajectory) -> tuple[str, str, str, int]:
    return item.dataset_id, item.task_id, item.run_id, item.step


def cohens_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa requires two non-empty label lists of equal length")
    observed = sum(left == right for left, right in zip(first, second)) / len(first)
    first_counts, second_counts = Counter(first), Counter(second)
    labels = set(first_counts) | set(second_counts)
    expected = sum(first_counts[label] / len(first) * second_counts[label] / len(second) for label in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def reconcile(first_path: str | Path, second_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    first, second = load_trajectories(first_path, True), load_trajectories(second_path, True)
    first_map, second_map = {_key(item): item for item in first}, {_key(item): item for item in second}
    if set(first_map) != set(second_map):
        raise DatasetValidationError("Annotation files do not contain identical trajectory-step keys")
    ordered_keys = sorted(first_map)
    binary_a = [str(first_map[key].actual_failure) for key in ordered_keys]
    binary_b = [str(second_map[key].actual_failure) for key in ordered_keys]
    category_a = [first_map[key].failure_type or "none" for key in ordered_keys]
    category_b = [second_map[key].failure_type or "none" for key in ordered_keys]
    conflicts = [key for key in ordered_keys if (first_map[key].actual_failure, first_map[key].failure_type) !=
                 (second_map[key].actual_failure, second_map[key].failure_type)]
    agreed = [first_map[key] for key in ordered_keys if key not in conflicts]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"agreement": output / "agreement.json", "agreed": output / "agreed.jsonl",
             "conflicts": output / "conflicts.csv", "adjudication": output / "adjudication.csv"}
    observed_binary = sum(a == b for a, b in zip(binary_a, binary_b)) / len(binary_a)
    observed_category = sum(a == b for a, b in zip(category_a, category_b)) / len(category_a)
    report = {"rows": len(ordered_keys), "agreements": len(agreed), "conflicts": len(conflicts),
              "binary_observed_agreement": observed_binary, "binary_cohens_kappa": cohens_kappa(binary_a, binary_b),
              "category_observed_agreement": observed_category,
              "category_cohens_kappa": cohens_kappa(category_a, category_b),
              "annotator_a": sorted({item.annotator_id for item in first}),
              "annotator_b": sorted({item.annotator_id for item in second})}
    paths["agreement"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_normalized_jsonl(agreed, paths["agreed"])
    fields = ["dataset_id", "task_id", "run_id", "step", "goal", "action", "observation",
              "annotator_a_failure", "annotator_a_type", "annotator_b_failure", "annotator_b_type"]
    adjudication_fields = fields + ["final_actual_failure", "final_failure_type", "adjudicator_id", "notes"]
    with paths["conflicts"].open("w", newline="", encoding="utf-8") as conflict_stream, \
            paths["adjudication"].open("w", newline="", encoding="utf-8") as adjudication_stream:
        conflict_writer = csv.DictWriter(conflict_stream, fieldnames=fields)
        adjudication_writer = csv.DictWriter(adjudication_stream, fieldnames=adjudication_fields)
        conflict_writer.writeheader()
        adjudication_writer.writeheader()
        for key in conflicts:
            left, right = first_map[key], second_map[key]
            row = {"dataset_id": left.dataset_id, "task_id": left.task_id, "run_id": left.run_id,
                   "step": left.step, "goal": left.goal, "action": left.action, "observation": left.observation,
                   "annotator_a_failure": left.actual_failure, "annotator_a_type": left.failure_type,
                   "annotator_b_failure": right.actual_failure, "annotator_b_type": right.failure_type}
            conflict_writer.writerow(row)
            adjudication_writer.writerow(row)
    print(f"Agreement rows={len(agreed)} conflicts={len(conflicts)} binary_kappa={report['binary_cohens_kappa']:.3f}")
    return paths


def resolve(first_path: str | Path, second_path: str | Path, adjudication_path: str | Path,
            output_path: str | Path) -> None:
    first, second = load_trajectories(first_path, True), load_trajectories(second_path, True)
    first_map, second_map = {_key(item): item for item in first}, {_key(item): item for item in second}
    final = []
    conflicts = set()
    for key in sorted(first_map):
        left, right = first_map[key], second_map[key]
        if (left.actual_failure, left.failure_type) == (right.actual_failure, right.failure_type):
            left.annotator_id = f"agreement:{left.annotator_id}+{right.annotator_id}"
            final.append(left)
        else:
            conflicts.add(key)
    decisions = {}
    with Path(adjudication_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset_id"], row["task_id"], row["run_id"], int(row["step"]))
            if not row.get("final_actual_failure", "").strip() or not row.get("final_failure_type", "").strip():
                raise DatasetValidationError(f"Conflict {key} has not been adjudicated")
            decisions[key] = row
    if set(decisions) != conflicts:
        raise DatasetValidationError("Adjudication rows do not exactly match annotation conflicts")
    for key, row in decisions.items():
        item = first_map[key]
        item.actual_failure = row["final_actual_failure"].strip().lower() in {"true", "1", "yes"}
        item.failure_type = row["final_failure_type"].strip()
        item.annotator_id = f"adjudicated:{row.get('adjudicator_id', '').strip()}"
        item.validate(f"adjudication {key}")
        final.append(item)
    write_normalized_jsonl(sorted(final, key=_key), output_path)
    print(f"Resolved {len(final)} labeled rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assign = commands.add_parser("assign")
    assign.add_argument("input", type=Path); assign.add_argument("output", type=Path)
    assign.add_argument("--annotator", required=True)
    compare = commands.add_parser("reconcile")
    compare.add_argument("first", type=Path); compare.add_argument("second", type=Path)
    compare.add_argument("--output-dir", type=Path, default=Path("data/collection/v08/review"))
    resolve_parser = commands.add_parser("resolve")
    resolve_parser.add_argument("first", type=Path); resolve_parser.add_argument("second", type=Path)
    resolve_parser.add_argument("adjudication", type=Path); resolve_parser.add_argument("output", type=Path)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("input", type=Path); bootstrap.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "assign":
        create_assignment(args.input, args.output, args.annotator)
        print(f"Assignment created: {args.output}")
    elif args.command == "bootstrap":
        bootstrap_labels(args.input, args.output)
    elif args.command == "reconcile":
        reconcile(args.first, args.second, args.output_dir)
    else:
        resolve(args.first, args.second, args.adjudication, args.output)


if __name__ == "__main__":
    try:
        main()
    except DatasetValidationError as exc:
        raise SystemExit(f"Annotation workflow stopped: {exc}") from None
