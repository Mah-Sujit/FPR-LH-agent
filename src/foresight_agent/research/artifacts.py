"""Generate checksums, manifests, documentation, audits, and simple SVG charts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from foresight_agent import __version__
from foresight_agent.data.external import contains_sensitive_text, load_trajectories
from foresight_agent.data.quality import quality_report


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict:
    names = ("foresight-agent", "numpy", "scipy", "scikit-learn", "joblib", "pytest", "openai")
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    packages["foresight-agent"] = __version__
    return {"python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(), "packages": packages}


def audit_run(dataset_path: Path, split_path: Path, metrics_path: Path) -> dict:
    items = load_trajectories(dataset_path, require_labels=True)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    train, validation, test = map(set, (
        splits["train_groups"], splits["validation_groups"], splits["test_groups"]))
    labels = {item.actual_failure for item in items}
    checks = {
        "dataset_valid": True,
        "labels_contain_both_classes": labels == {False, True},
        "train_validation_disjoint": train.isdisjoint(validation),
        "train_test_disjoint": train.isdisjoint(test),
        "validation_test_disjoint": validation.isdisjoint(test),
        "metrics_present": metrics_path.exists() and metrics_path.stat().st_size > 0,
        "observations_redacted": all(not contains_sensitive_text(item.observation) for item in items),
    }
    return {"passed": all(checks.values()), "checks": checks, "quality": quality_report(items)}


def write_model_card(metadata_path: Path, metrics_path: Path, dataset_hash: str, output: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with metrics_path.open(encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    selected = next(row for row in metrics if row["model"] == metadata["selected_model"])
    text = f"""# ForesightAgent Model Card

## Model details

- Model: {metadata['selected_model']}
- Version: {metadata['version']}
- Decision threshold: {float(metadata['threshold']):.2f}
- Dataset SHA-256: `{dataset_hash}`
- Features: {', '.join(metadata['features'])}

## Intended use

Research baseline for estimating whether an agent trajectory step is likely to fail and triggering a recovery policy. It is not intended for safety-critical, medical, legal, financial, or autonomous high-impact decisions.

## Held-out results

- Precision: {float(selected['precision']):.3f}
- Recall: {float(selected['recall']):.3f}
- F1: {float(selected['f1']):.3f}
- ROC-AUC: {float(selected['roc_auc']):.3f}
- PR-AUC: {float(selected['pr_auc']):.3f}
- Brier score: {float(selected['brier']):.3f}
- Calibration error: {float(selected['calibration_error']):.3f}

## Limitations

Results depend on task selection and label quality. Bootstrap labels are rules, not independent human judgments. Performance must be re-established on representative external trajectories before deployment.
"""
    output.write_text(text, encoding="utf-8")


def write_datasheet(dataset_path: Path, manifest: dict, output: Path) -> None:
    items = load_trajectories(dataset_path, require_labels=True)
    quality = quality_report(items)
    text = f"""# ForesightAgent Dataset Datasheet

## Composition

- Dataset ID: {items[0].dataset_id}
- Rows: {quality['rows']}
- Tasks: {quality['tasks']}
- Runs: {quality['runs']}
- Label counts: {json.dumps(quality['label_counts'], sort_keys=True)}
- Failure types: {json.dumps(quality['failure_type_counts'], sort_keys=True)}
- Annotators: {quality['annotators']}

## Provenance

- Source: {manifest.get('source', 'not supplied')}
- Catalog version: {manifest.get('version', 'not supplied')}
- Network used during collection: {manifest.get('network_used', 'unknown')}
- Label mode: {manifest.get('label_mode', 'unknown')}

## Collection and processing

Trajectories were produced by explicitly allow-listed local tools. Normalization validates identifiers, ranges, labels, and duplicate step keys and redacts common API-key and email patterns.

## Recommended uses and limitations

Suitable for pipeline tests and controlled failure-recovery studies. The provisional bootstrap labels are not suitable as evidence of human agreement or real-world generalization. Users remain responsible for consent, licensing, privacy review, and representative sampling.
"""
    output.write_text(text, encoding="utf-8")


def write_metrics_svg(metrics_path: Path, output: Path) -> None:
    with metrics_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    width, height, margin = 760, 380, 70
    chart_height = height - 2 * margin
    colors = ("#2563eb", "#059669", "#d97706")
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Held-out failure-prediction metrics</text>',
                f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>']
    group_width = (width - 2 * margin) / len(rows)
    for index, row in enumerate(rows):
        base_x = margin + index * group_width + 20
        for metric_index, metric in enumerate(("precision", "recall", "f1")):
            value = float(row[metric]); bar_width = 28
            x = base_x + metric_index * 36; bar_height = value * chart_height
            y = height - margin - bar_height
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[metric_index]}"/>')
        elements.append(f'<text x="{base_x+46:.1f}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{row["model"]}</text>')
    for index, (metric, color) in enumerate(zip(("Precision", "Recall", "F1"), colors)):
        x = 235 + index * 115
        elements.extend([f'<rect x="{x}" y="350" width="14" height="14" fill="{color}"/>',
                         f'<text x="{x+20}" y="362" font-family="sans-serif" font-size="12">{metric}</text>'])
    elements.append("</svg>")
    output.write_text("\n".join(elements), encoding="utf-8")


def write_checksums(root: Path, output: Path) -> None:
    entries = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
               if path.is_file() and path != output}
    output.write_text(json.dumps({"algorithm": "SHA-256", "files": entries}, indent=2), encoding="utf-8")
