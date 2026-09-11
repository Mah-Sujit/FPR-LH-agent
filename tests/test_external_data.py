import csv
import json

import pytest

from foresight_agent.config import ExperimentConfig
from foresight_agent.data.external import DatasetValidationError, contains_sensitive_text, load_trajectories
from foresight_agent.data.quality import quality_report, write_annotation_template, write_quality_report
from foresight_agent.evaluation.external_training import train_external


def row(task, step=1, failure=None, failure_type=None):
    return {"dataset_id": "study", "task_id": task, "run_id": "run-1", "step": step,
            "goal": "Research user@example.com", "action": "search",
            "observation": "No secret sk-example-123456789012 should remain",
            "success": not bool(failure), "confidence": 0.2 if failure else 0.9,
            "progress_score": 0.3 if failure else 0.8,
            "actual_failure": failure, "failure_type": failure_type}


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")


def test_external_loader_redacts_and_templates(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [row("task-1")])
    items = load_trajectories(source)
    assert "user@example.com" not in items[0].goal
    assert "sk-example" not in items[0].observation
    assert quality_report(items)["redacted_rows"] == 1
    template = tmp_path / "labels.csv"
    write_annotation_template(items, template)
    with template.open(encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert saved["actual_failure"] == ""
    report = tmp_path / "nested" / "quality.json"
    write_quality_report(items, report)
    assert report.exists()
    assert contains_sensitive_text("task-grouped split") is False
    assert contains_sensitive_text("sk-example-123456789012") is True


def test_external_loader_rejects_inconsistent_labels(tmp_path):
    source = tmp_path / "bad.jsonl"
    write_jsonl(source, [row("task-1", failure=True, failure_type=None)])
    with pytest.raises(DatasetValidationError, match="requires a failure_type"):
        load_trajectories(source, require_labels=True)


def test_external_loader_names_unlabeled_rows(tmp_path):
    source = tmp_path / "unlabeled.jsonl"
    write_jsonl(source, [row("task-7")])
    with pytest.raises(DatasetValidationError, match="task-7/run-1/step-1"):
        load_trajectories(source, require_labels=True)


def test_external_training_uses_disjoint_task_groups(tmp_path):
    source = tmp_path / "labeled.jsonl"
    rows = []
    for index in range(18):
        task = f"task-{index:02d}"
        rows.extend([row(task, 1, False, "none"), row(task, 2, True, "hallucination")])
    write_jsonl(source, rows)
    paths = train_external(source, tmp_path / "results", ExperimentConfig(seed=5))
    assert all(path.exists() for path in paths.values())
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    train, validation, test = map(set, (
        splits["train_groups"], splits["validation_groups"], splits["test_groups"]))
    assert train.isdisjoint(validation | test)
    assert validation.isdisjoint(test)
