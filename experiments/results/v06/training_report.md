# ForesightAgent Learned Predictor Study

Selected model: **random_forest**  
Validation-selected threshold: **0.50**  
Split: task-grouped train/validation/test (no task appears in multiple splits)

## Held-out prediction results

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rule_based | 0.05 | 0.825 | 0.868 | 0.846 | 0.916 | 0.920 | 0.150 | 0.204 |
| logistic_regression | 0.25 | 0.867 | 0.947 | 0.906 | 0.970 | 0.973 | 0.051 | 0.043 |
| decision_tree | 0.30 | 0.860 | 0.974 | 0.914 | 0.968 | 0.937 | 0.053 | 0.068 |
| random_forest | 0.50 | 0.914 | 0.974 | 0.943 | 0.982 | 0.986 | 0.036 | 0.063 |

## Recovery on held-out task groups

| Policy | Task success | Average steps |
|---|---:|---:|
| no_recovery | 57.5% | 1.00 |
| rule_based | 82.5% | 1.30 |
| learned | 96.7% | 1.57 |

Models are trained only on seeded synthetic trajectories. External validation is required.
