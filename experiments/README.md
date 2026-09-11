# Experiments

Use this directory for evaluation scripts and experiment configurations. Keep core agent behavior inside `src/foresight_agent` so experiments remain thin and reproducible.

Suggested Version 0.2 measurements:

- end-to-end task success
- failure-prediction precision, recall, and calibration
- warning lead time before terminal failure
- recovery success by action and failure type
- unnecessary intervention rate
- steps, latency, and model/tool cost

Run the included comparison with `foresight-evaluate`. Generated CSV files are written under `results/` and ignored by Git. The committed `config.json` records the baseline thresholds, budgets, and risk weights.

Run the 24-task repeated offline study with `foresight-study` or `python3 -m foresight_agent.evaluation.study`. It produces both raw CSV data and a Markdown report under `results/v04/`.

Run the noisy Version 0.5 experiment with `foresight-noisy-study` or `python3 -m foresight_agent.evaluation.noisy_study`. Its `results/v05/` artifacts include raw runs, threshold sweeps with confusion counts, calibration bins, and a confidence-interval report.

Train the Version 0.6 predictors with `foresight-train` or `python3 -m foresight_agent.evaluation.training`. The `results/v06/` directory receives the calibrated model, feature metadata, leakage-free split manifest, held-out metrics, and recovery comparison.

Version 0.7 accepts external JSONL or CSV trajectories through `foresight-data`, then trains with `foresight-train-external`. See `data/external/SCHEMA.md`; external datasets require at least 15 distinct labeled task groups.

Version 0.8 collects a 20-task offline-tool dataset with `foresight-collect`, then supports two independent reviewers, Cohen’s kappa, conflict adjudication, and final label resolution through `foresight-annotations`.

Version 0.9 packages the complete reproducibility record with `foresight-package` or `python3 -m foresight_agent.research.pipeline`. Every timestamped run contains checksums, audits, frozen configuration/environment details, model/data cards, raw metrics, and an SVG chart.
