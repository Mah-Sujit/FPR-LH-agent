# ForesightAgent Dataset Datasheet

## Composition

- Dataset ID: fprlh-offline-tools-v1
- Rows: 40
- Tasks: 20
- Runs: 20
- Label counts: {"false": 32, "true": 8}
- Failure types: {"missing_evidence": 2, "none": 32, "tool_execution_error": 6}
- Annotators: 1

## Provenance

- Source: not supplied
- Catalog version: not supplied
- Network used during collection: unknown
- Label mode: human-reviewed

## Collection and processing

Trajectories were produced by explicitly allow-listed local tools. Normalization validates identifiers, ranges, labels, and duplicate step keys and redacts common API-key and email patterns.

## Recommended uses and limitations

Suitable for pipeline tests and controlled failure-recovery studies. The provisional bootstrap labels are not suitable as evidence of human agreement or real-world generalization. Users remain responsible for consent, licensing, privacy review, and representative sampling.
