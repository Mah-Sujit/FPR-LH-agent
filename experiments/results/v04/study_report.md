# ForesightAgent Offline Study

Tasks per policy/run: 24  
Repeated runs: 3  
Base seed: 42

## Overall results

| Policy | Success rate | Avg. steps | Avg. interventions | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| no_recovery | 12.5% | 1.25 | 0.00 | 1.000 | 1.000 | 1.000 |
| rule_based | 100.0% | 2.88 | 1.38 | 1.000 | 1.000 | 1.000 |
| oracle | 100.0% | 2.75 | 1.38 | 1.000 | 1.000 | 1.000 |

## Success by failure type

| Failure type | No recovery | Rule-based | Oracle |
|---|---:|---:|---:|
| contradiction | 0.0% | 100.0% | 100.0% |
| hallucination | 0.0% | 100.0% | 100.0% |
| missing_evidence | 0.0% | 100.0% | 100.0% |
| none | 100.0% | 100.0% | 100.0% |
| planning_error | 0.0% | 100.0% | 100.0% |
| repetition_loop | 0.0% | 100.0% | 100.0% |
| tool_execution_error | 0.0% | 100.0% | 100.0% |
| tool_selection_error | 0.0% | 100.0% | 100.0% |

This synthetic study validates experimental plumbing and policy behavior. It does not establish real-world generalization.
