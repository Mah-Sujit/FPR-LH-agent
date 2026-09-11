# ForesightAgent Model Card

## Model details

- Model: logistic_regression
- Version: 0.7.0
- Decision threshold: 0.05
- Dataset SHA-256: `85ddc3d12fd1a3d43b6d44d97c293db4dea28f63ca5071a508a6574b9aacace1`
- Features: confidence, tool_success, progress_score, progress_delta, repeated_action, contradiction, retry_count, step_position, previous_risk, signal_tool_error, signal_low_confidence, signal_repetition, signal_contradiction, signal_no_progress

## Intended use

Research baseline for estimating whether an agent trajectory step is likely to fail and triggering a recovery policy. It is not intended for safety-critical, medical, legal, financial, or autonomous high-impact decisions.

## Held-out results

- Precision: 0.250
- Recall: 1.000
- F1: 0.400
- ROC-AUC: 1.000
- PR-AUC: 1.000
- Brier score: 0.032
- Calibration error: 0.157

## Limitations

Results depend on task selection and label quality. Bootstrap labels are rules, not independent human judgments. Performance must be re-established on representative external trajectories before deployment.
