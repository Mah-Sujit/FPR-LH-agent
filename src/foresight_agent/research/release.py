"""Create a final-report draft and enforce the human-evidence release gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from foresight_agent import __version__
from foresight_agent.data.external import DatasetValidationError, load_trajectories


def evaluate_release_gate(package: str | Path, dataset: str | Path,
                          agreement: str | Path | None = None) -> dict:
    package, dataset = Path(package), Path(dataset)
    items = load_trajectories(dataset, require_labels=True)
    audit_path = package / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"passed": False}
    non_human_markers = ("automated", "bootstrap", "provisional", "codex", "assisted")
    automated = any(
        any(marker in f"{item.annotator_id or ''} {item.source or ''}".lower()
            for marker in non_human_markers)
        for item in items
    )
    agreement_data = {}
    if agreement and Path(agreement).exists():
        agreement_data = json.loads(Path(agreement).read_text(encoding="utf-8"))
    reviewer_a = set(agreement_data.get("annotator_a", [])) - {None, ""}
    reviewer_b = set(agreement_data.get("annotator_b", [])) - {None, ""}
    checks = {
        "package_audit_passed": bool(audit.get("passed")),
        "minimum_15_task_groups": len({(item.dataset_id, item.task_id) for item in items}) >= 15,
        "both_label_classes_present": {item.actual_failure for item in items} == {False, True},
        "no_bootstrap_labels": not automated,
        "agreement_report_present": bool(agreement_data),
        "two_distinct_reviewers": bool(reviewer_a and reviewer_b and reviewer_a.isdisjoint(reviewer_b)),
        "binary_kappa_at_least_0_60": float(agreement_data.get("binary_cohens_kappa", -1)) >= 0.60,
        "all_conflicts_adjudicated": len(items) == int(agreement_data.get("rows", -1)),
    }
    return {"ready_for_1_0": all(checks.values()), "project_version": __version__,
            "dataset": str(dataset), "package": str(package), "checks": checks,
            "task_groups": len({(item.dataset_id, item.task_id) for item in items}),
            "rows": len(items), "binary_kappa": agreement_data.get("binary_cohens_kappa")}


def write_release_draft(package: str | Path, dataset: str | Path, output: str | Path) -> Path:
    package, dataset, output = Path(package), Path(dataset), Path(output)
    manifest = json.loads((package / "run_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
    metadata = json.loads((package / "model" / "external_metadata.json").read_text(encoding="utf-8"))
    with (package / "model" / "external_metrics.csv").open(encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    items = load_trajectories(dataset, require_labels=True)
    provisional = any("automated-bootstrap" in (item.annotator_id or "") for item in items)
    status = "PROVISIONAL — automated bootstrap labels" if provisional else "HUMAN-LABELED CANDIDATE"
    lines = ["# ForesightAgent Final Research Report — Draft", "", f"**Evidence status:** {status}", "",
             "## Abstract", "", "ForesightAgent predicts step-level failure risk during long-horizon agent execution and selects among continue, verify, retry, replan, rollback, and abstain recovery actions. This report summarizes the modular implementation and its held-out evaluation.", "",
             "## System", "", "The system separates planning, execution, trajectory monitoring, feature extraction, risk prediction, recovery control, and evaluation. Both transparent weighted rules and calibrated learned predictors implement a shared scoring boundary.", "",
             "## Dataset and protocol", "", f"The packaged dataset contains {len(items)} steps across {len({item.task_id for item in items})} task groups. Tasks are split as whole groups across train, validation, and test partitions. Threshold selection uses validation tasks only. Dataset SHA-256: `{manifest['dataset_sha256']}`.", "",
             "## Held-out prediction results", "", "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['model']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} | {float(row['roc_auc']):.3f} | {float(row['pr_auc']):.3f} | {float(row['brier']):.3f} | {float(row['calibration_error']):.3f} |")
    lines.extend(["", "## Selected predictor", "", f"The validation procedure selected **{metadata['selected_model']}** at threshold **{float(metadata['threshold']):.2f}**.", "",
                  "## Reproducibility", "", f"Packaged run: `{manifest['run_id']}`. Audit passed: **{audit['passed']}**. The package includes frozen configuration, environment metadata, task splits, model/data cards, raw metrics, visualization, and SHA-256 checksums.", "",
                  "## Limitations", "", "Synthetic and bootstrap-labeled results do not establish performance on naturally occurring agent failures. Small samples, subjective labels, distribution shift, and tool/provider changes limit generalization. Final claims require two independent human reviewers and adjudicated labels.", "",
                  "## Conclusion", "", "The software and evaluation pipeline are complete as a release candidate. Final Version 1.0 evidence remains conditional on the human-annotation release gate."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _latest_package(root: Path) -> Path:
    packages = sorted(path for path in root.iterdir() if path.is_dir())
    if not packages:
        raise DatasetValidationError(f"No packaged runs found under {root}")
    return packages[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft")
    draft.add_argument("--package", type=Path); draft.add_argument("--dataset", type=Path, required=True)
    draft.add_argument("--output", type=Path, default=Path("docs/FINAL_REPORT_DRAFT.md"))
    gate = commands.add_parser("gate")
    gate.add_argument("--package", type=Path); gate.add_argument("--dataset", type=Path, required=True)
    gate.add_argument("--agreement", type=Path)
    args = parser.parse_args()
    package = args.package or _latest_package(Path("experiments/results/v09"))
    if args.command == "draft":
        print(f"Report draft: {write_release_draft(package, args.dataset, args.output)}")
    else:
        result = evaluate_release_gate(package, args.dataset, args.agreement)
        print(json.dumps(result, indent=2))
        if not result["ready_for_1_0"]:
            raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (DatasetValidationError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Release check stopped: {exc}") from None
