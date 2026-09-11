"""Prepare, validate, summarize, and template external trajectory data."""

import argparse
import json
from pathlib import Path

from .external import DatasetValidationError, load_trajectories, write_normalized_jsonl
from .quality import quality_report, write_annotation_template, write_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("input", type=Path)
    validate.add_argument("--require-labels", action="store_true")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--report", type=Path)
    template = subparsers.add_parser("template")
    template.add_argument("input", type=Path)
    template.add_argument("output", type=Path)
    args = parser.parse_args()
    items = load_trajectories(args.input, getattr(args, "require_labels", False))
    if args.command == "validate":
        print(json.dumps(quality_report(items), indent=2))
    elif args.command == "prepare":
        write_normalized_jsonl(items, args.output)
        if args.report:
            write_quality_report(items, args.report)
        print(f"Prepared {len(items)} rows: {args.output}")
    else:
        write_annotation_template(items, args.output)
        print(f"Annotation template: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except DatasetValidationError as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from None
