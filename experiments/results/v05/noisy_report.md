# ForesightAgent Noisy Study

Tasks: 30 × 10 runs per policy  
Base seed: 42

## Policy comparison

| Policy | Task success (95% CI) | Avg steps | Avg interventions | Avg cost | Precision | Recall | F1 | Unnecessary interventions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_recovery | 59.7% (54.0%–65.1%) | 1.00 | 0.00 | 1.00 | 1.000 | 0.438 | 0.609 | 0.0% |
| rule_based | 76.0% (70.9%–80.5%) | 1.18 | 0.18 | 1.27 | 1.000 | 0.433 | 0.604 | 0.0% |
| oracle | 100.0% (98.7%–100.0%) | 1.46 | 0.46 | 1.69 | 0.982 | 0.409 | 0.577 | 0.0% |

## Threshold sweep

Best observed F1 threshold: **0.10** (F1 0.709).

Raw confusion matrices, calibration bins, and threshold results are included as CSV files.

Simulated latency and cost are comparative units, not provider billing estimates. Results measure seeded synthetic noise and do not establish external validity.
