# External Trajectory Evaluation

Rows: 40  
Task groups: 20  
Selected model: **logistic_regression**  
Validation-selected threshold: **0.05**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.250 | 1.000 | 0.400 | 1.000 | 1.000 | 0.032 | 0.157 |
| decision_tree | 0.250 | 1.000 | 0.400 | 0.833 | 0.500 | 0.180 | 0.231 |
| random_forest | 0.250 | 1.000 | 0.400 | 0.833 | 0.500 | 0.160 | 0.183 |

Metrics are valid only to the extent that task sampling and human labels are reliable.
