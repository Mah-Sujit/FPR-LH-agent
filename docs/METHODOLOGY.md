# ForesightAgent research methodology

## Research question

Can an agent estimate impending trajectory failure and select a recovery action that improves long-horizon task success at acceptable cost?

## Experimental units

The primary unit is a task run. Step-level records support prediction metrics, while task-level outcomes measure recovery utility. Repeated executions must record seeds and configuration snapshots.

## Data partitions

Split by task identity. No task may occur across train, validation, and test partitions. Fit models on training tasks, select thresholds on validation tasks, and report final metrics once on held-out test tasks.

## Labels

`actual_failure` indicates whether a step is incorrect or threatens task completion. Failure types follow the documented taxonomy. Credible external evaluation requires two independent annotators, agreement reporting, and adjudication of conflicts. Automated bootstrap labels are restricted to pipeline development.

## Comparisons

Compare no recovery, rule-based prediction, learned prediction, and an oracle upper bound. Report prediction quality, calibration, task success, interventions, steps, latency, and cost.

## Statistical reporting

Report precision, recall, F1, ROC-AUC, PR-AUC, Brier score, calibration error, confusion counts, and task-success confidence intervals. Preserve per-run rows so uncertainty can be recomputed.

## Reproducibility

Every packaged run includes a unique run ID, configuration snapshot, environment manifest, task-group split, data/model metadata, SHA-256 checksums, audit results, raw metric tables, and a model card and dataset datasheet.

## Threats to validity

Synthetic failures may not represent real agent behavior. Tool-reported success is not ground truth. Small or imbalanced datasets can destabilize calibration and threshold selection. Repeated variants of the same task can create hidden leakage. Human labels can be subjective. Provider, prompt, and tool changes can cause distribution shift.
