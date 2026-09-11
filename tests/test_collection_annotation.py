import csv
import json

from foresight_agent.data.annotation import bootstrap_labels, create_assignment, reconcile, resolve
from foresight_agent.data.collector import collect, load_catalog
from foresight_agent.data.external import load_trajectories


def _label(path, labels):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream)); fields = rows[0].keys()
    for row, (failure, kind) in zip(rows, labels):
        row["actual_failure"], row["failure_type"] = str(failure), kind
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_catalog_collects_20_tasks_and_manifest(tmp_path):
    metadata, tasks = load_catalog("experiments/tasks/v08_tasks.json")
    assert metadata["version"] == "1.0.0"
    assert len(tasks) == 20
    paths = collect("experiments/tasks/v08_tasks.json", tmp_path, "data/collection")
    items = load_trajectories(paths["trajectories"])
    assert len(items) == 40
    assert len({item.task_id for item in items}) == 20
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["network_used"] is False
    labeled = tmp_path / "provisional.jsonl"
    bootstrap_labels(paths["trajectories"], labeled)
    labels = load_trajectories(labeled, require_labels=True)
    assert len(labels) == 40
    assert any(item.failure_type == "missing_evidence" for item in labels)
    assert any(item.failure_type == "tool_execution_error" for item in labels)
    assert all(item.annotator_id == "automated-bootstrap-v0.8" for item in labels)
    labeled_csv = tmp_path / "provisional.csv"
    bootstrap_labels(paths["trajectories"], labeled_csv)
    assert len(load_trajectories(labeled_csv, require_labels=True)) == 40


def test_dual_annotation_reconcile_and_resolve(tmp_path):
    collected = collect("experiments/tasks/v08_tasks.json", tmp_path / "raw", "data/collection")
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    create_assignment(collected["trajectories"], first, "reviewer-a")
    create_assignment(collected["trajectories"], second, "reviewer-b")
    base = [(False, "none")] * 40
    alternate = list(base); alternate[0] = (True, "other")
    _label(first, base); _label(second, alternate)
    paths = reconcile(first, second, tmp_path / "review")
    report = json.loads(paths["agreement"].read_text(encoding="utf-8"))
    assert report["conflicts"] == 1

    with paths["adjudication"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream)); fields = rows[0].keys()
    rows[0]["final_actual_failure"] = "False"
    rows[0]["final_failure_type"] = "none"
    rows[0]["adjudicator_id"] = "lead-reviewer"
    with paths["adjudication"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    final = tmp_path / "final.jsonl"
    resolve(first, second, paths["adjudication"], final)
    assert len(load_trajectories(final, require_labels=True)) == 40
