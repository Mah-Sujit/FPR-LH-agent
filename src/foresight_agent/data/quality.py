"""Dataset-quality summaries and annotation templates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .external import ExternalTrajectory


def quality_report(items: list[ExternalTrajectory]) -> dict:
    labels = Counter("unlabeled" if item.actual_failure is None else str(item.actual_failure).lower()
                     for item in items)
    failure_types = Counter(item.failure_type or "unlabeled" for item in items)
    return {"rows": len(items), "datasets": len({item.dataset_id for item in items}),
            "tasks": len({(item.dataset_id, item.task_id) for item in items}),
            "runs": len({(item.dataset_id, item.task_id, item.run_id) for item in items}),
            "label_counts": dict(sorted(labels.items())),
            "failure_type_counts": dict(sorted(failure_types.items())),
            "annotators": len({item.annotator_id for item in items if item.annotator_id}),
            "redacted_rows": sum("[REDACTED_" in item.goal or "[REDACTED_" in item.observation for item in items)}


def write_quality_report(items: list[ExternalTrajectory], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(quality_report(items), indent=2), encoding="utf-8")


def write_annotation_template(items: list[ExternalTrajectory], path: str | Path) -> None:
    fields = list(asdict(items[0]))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = asdict(item)
            row["actual_failure"] = "" if item.actual_failure is None else item.actual_failure
            row["failure_type"] = item.failure_type or ""
            writer.writerow(row)
