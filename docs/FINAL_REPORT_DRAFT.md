# ForesightAgent Final Research Report — Draft

**Evidence status:** PROVISIONAL — automated bootstrap labels

## Abstract

ForesightAgent predicts step-level failure risk during long-horizon agent execution and selects among continue, verify, retry, replan, rollback, and abstain recovery actions. This report summarizes the modular implementation and its held-out evaluation.

## System

The system separates planning, execution, trajectory monitoring, feature extraction, risk prediction, recovery control, and evaluation. Both transparent weighted rules and calibrated learned predictors implement a shared scoring boundary.

## Dataset and protocol

The packaged dataset contains 40 steps across 20 task groups. Tasks are split as whole groups across train, validation, and test partitions. Threshold selection uses validation tasks only. Dataset SHA-256: `85ddc3d12fd1a3d43b6d44d97c293db4dea28f63ca5071a508a6574b9aacace1`.

## Held-out prediction results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.250 | 1.000 | 0.400 | 1.000 | 1.000 | 0.032 | 0.157 |
| decision_tree | 0.250 | 1.000 | 0.400 | 0.833 | 0.500 | 0.180 | 0.231 |
| random_forest | 0.250 | 1.000 | 0.400 | 0.833 | 0.500 | 0.160 | 0.183 |

## Selected predictor

The validation procedure selected **logistic_regression** at threshold **0.05**.

## Reproducibility

Packaged run: `20260911T233408662072Z-85ddc3d1`. Audit passed: **True**. The package includes frozen configuration, environment metadata, task splits, model/data cards, raw metrics, visualization, and SHA-256 checksums.

## Limitations

Synthetic and bootstrap-labeled results do not establish performance on naturally occurring agent failures. Small samples, subjective labels, distribution shift, and tool/provider changes limit generalization. Final claims require two independent human reviewers and adjudicated labels.

## Conclusion

The software and evaluation pipeline are complete as a release candidate. Final Version 1.0 evidence remains conditional on the human-annotation release gate.
