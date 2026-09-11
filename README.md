# ForesightAgent

**Version 1.0.0 release candidate 1** is a reproducible research system for predictive failure recovery, offline trajectory collection, annotation, learned-model evaluation, auditable experiment packaging, and evidence-gated release preparation.

It deliberately uses plain Python and transparent rules. There is no model provider, framework, API key, or network dependency. This makes the control loop easy to understand, test, and later compare against learned failure predictors.

## What it contains

The single-agent loop performs the following cycle for every plan step:

```text
plan -> execute -> record trajectory -> detect signals -> score risk
     -> choose recovery -> update state -> repeat
```

The recovery controller supports:

| Action | Meaning |
|---|---|
| `CONTINUE` | Accept the result and advance to the next step. |
| `VERIFY` | Mark a moderate-risk step for another pass before accepting it. |
| `RETRY` | Repeat a failed step up to the retry budget. |
| `REPLAN` | Ask the planner to replace or revise the plan. |
| `ROLLBACK` | Restore the latest successful in-memory snapshot. |
| `ABSTAIN` | Stop safely when automated recovery is exhausted. |

Version 0.1 detects five explainable signals: action/tool error, low confidence, repeated action, contradiction, and lack of progress. A weighted scorer converts them into a risk value from 0 to 1.

## Repository layout

```text
src/foresight_agent/
├── agent/          # state, planner, and execution loop
├── monitoring/     # trajectory logger and failure signals
├── prediction/     # risk scoring
├── recovery/       # controller and state transitions
└── example.py      # runnable demonstration
tests/              # focused unit and integration tests
data/trajectories/  # generated JSONL run logs (git-ignored)
experiments/         # configuration and generated benchmark results
src/foresight_agent/llm/    # provider-neutral, mock, and optional OpenAI clients
src/foresight_agent/tools/  # safe calculator and local-text tools
```

## Run locally on macOS

You need Python 3.10 or newer. In Terminal, change into this repository and run:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
pytest
python3 -m foresight_agent.example
```

The example intentionally fails once, selects `RETRY`, recovers, and finishes. Its full trajectory is written to `data/trajectories/example.jsonl`. You can also run it through the installed command:

```bash
foresight-example
```

## Run the evaluation benchmark

Compare no recovery, rule-based recovery, and an oracle upper-bound policy:

```bash
foresight-evaluate
```

Alternatively:

```bash
python3 -m foresight_agent.evaluation.runner
```

The runner executes deterministic healthy, transient-failure, uncertainty, and persistent-failure cases. It reports task success and prediction precision/recall/F1, then writes per-run results to `experiments/results/benchmark.csv`.

Edit `experiments/config.json` to change risk thresholds, recovery budgets, iteration limits, or signal weights. You can use a different configuration or output file with:

```bash
foresight-evaluate --config experiments/config.json --output experiments/results/my-run.csv
```

To leave the virtual environment, run `deactivate`.

## Create your own task

Each plan step is a Python function that receives `AgentState` and returns observation metadata:

```python
from foresight_agent import AgentExecutor, TaskStep
from foresight_agent.agent.planner import Planner

def inspect_data(state):
    state.memory["rows"] = 42
    return {
        "observation": "Found 42 valid rows",
        "success": True,
        "confidence": 0.9,
        "progress_score": 0.5,
    }

steps = [TaskStep("inspect_data", inspect_data)]
result = AgentExecutor(Planner(steps)).run("Inspect the dataset")
print(result.status)
```

A step may additionally return `repeated_action`, `contradiction`, or `error`. Unexpected Python exceptions are caught and retained in the trajectory rather than crashing the run.

## Research extension points

- Replace `FailureSignalDetector` with richer trajectory features.
- Replace `RiskScorer` with a calibrated classifier while retaining the same interface.
- Add a planner `replan_hook` for domain-specific plan repair.
- Replace step functions with real tools or LLM calls.
- Evaluate task success, early-warning lead time, calibration, recovery success, cost, and unnecessary intervention rate.
- Store experiment configuration and random seeds alongside trajectories for reproducibility.

The current rules and benchmarks are a baseline, not a claim of learned failure prediction. The oracle is an experimental upper bound because it can inspect ground-truth labels; it is not a deployable policy.

## Version 0.3 structured LLM planner

Run the offline example first. It uses a deterministic mock model and needs no API key:

```bash
python3 -m foresight_agent.llm_example
```

The model returns a JSON plan containing named steps, registered tools, and arguments. Tool names are allow-listed. The calculator accepts arithmetic only, while local text tools resolve every path under an explicitly configured root.

### Optional real OpenAI run

Install the optional client and provide the key only through your shell environment:

```bash
python3 -m pip install -e ".[dev,llm]"
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gpt-5-mini"
python3 -m foresight_agent.openai_example "Calculate 15 percent of 240"
```

If setting an environment variable is inconvenient, unset it and run the same command. The example will securely prompt for the key; pasted characters are intentionally hidden:

```bash
unset OPENAI_API_KEY
python3 -m foresight_agent.openai_example "Calculate 15 percent of 240"
```

Do not put the key in source code or commit it. `.env` is ignored as an additional safeguard. Real-model prompt, response, token, model, and latency telemetry is written to `data/telemetry/openai.jsonl`; cost is intentionally left unset because pricing changes and should be supplied by an experiment-specific pricing table.

The architecture accepts another provider by implementing the single `LLMClient.generate(system_prompt, user_prompt)` method. Tests use `MockLLMClient`, so continuous integration remains deterministic and free of API charges.

## Version 0.4 offline study

Run the larger study without an API key:

```bash
python3 -m foresight_agent.evaluation.study
```

The study contains 24 labeled tasks covering healthy execution, tool execution errors, tool selection errors, planning errors, repetition loops, contradictions, hallucinations, and missing evidence. By default, every policy runs every task three times using recorded seeds.

Outputs:

- `experiments/results/v04/study_runs.csv` contains one row per task run.
- `experiments/results/v04/study_report.md` summarizes policy success, prediction metrics, steps, interventions, and success by failure type.

Change `runs_per_task` and `seed` in `experiments/config.json` to control repeated experiments. These are synthetic failure-injection tasks intended to validate methodology; real-world claims require an external task dataset and human-checked labels.

## Version 0.5 noisy study

Run the probabilistic evaluation without an API key:

```bash
python3 -m foresight_agent.evaluation.noisy_study
```

This study evaluates 30 tasks over multiple seeds. Failure signals deliberately overlap: some failures are weakly signaled, some successes produce false alarms, and recovery attempts can fail. It produces:

- `experiments/results/v05/noisy_runs.csv`
- `experiments/results/v05/threshold_sweep.csv`
- `experiments/results/v05/calibration.csv`
- `experiments/results/v05/noisy_report.md`

The report includes 95% Wilson confidence intervals, task success, precision/recall/F1, extra steps, interventions, unnecessary intervention rate, simulated latency, and simulated cost. Threshold CSV rows include confusion-matrix counts plus true-positive and false-positive rates for ROC analysis. Adjust `noisy_runs_per_task`, `seed`, and `threshold_sweep` in `experiments/config.json`.

## Version 0.6 learned failure prediction

Install the optional machine-learning dependencies and train the models:

```bash
python3 -m pip install -e ".[dev,ml]"
python3 -m foresight_agent.evaluation.training
```

The pipeline extracts a stable numerical feature vector from every trajectory step and splits data by task identity. Entire tasks—not individual rows—are assigned to train, validation, or test partitions to prevent leakage. It trains and calibrates logistic regression, decision tree, and random forest models. Decision thresholds are selected using validation data only and metrics are reported on held-out task groups.

Generated artifacts:

- `experiments/results/v06/best_model.joblib`
- `experiments/results/v06/model_metadata.json`
- `experiments/results/v06/model_metrics.csv`
- `experiments/results/v06/split_manifest.json`
- `experiments/results/v06/recovery_comparison.csv`
- `experiments/results/v06/training_report.md`

The report compares precision, recall, F1, ROC-AUC, PR-AUC, Brier score, calibration error, and downstream recovery success. Choose the runtime scorer through `experiments/config.json`:

```json
{
  "predictor": "rule_based"
}
```

or use the saved model:

```json
{
  "predictor": "learned",
  "model_path": "experiments/results/v06/best_model.joblib"
}
```

Synthetic training results are methodological baselines and must not be treated as evidence of performance on real agent trajectories.

## Version 0.7 external trajectory evaluation

The interchange format is documented in `data/external/SCHEMA.md`. Prepare an annotation sheet from collected JSONL or CSV data:

```bash
python3 -m foresight_agent.data.cli template \
  data/external/example_unlabeled.jsonl \
  data/external/annotations.csv
```

Complete `actual_failure`, `failure_type`, and preferably `annotator_id`, then validate it:

```bash
python3 -m foresight_agent.data.cli validate \
  data/external/annotations.csv --require-labels
```

Normalize and redact common secret/email patterns while writing a quality report:

```bash
python3 -m foresight_agent.data.cli prepare \
  data/external/annotations.csv \
  data/external/normalized.jsonl \
  --report experiments/results/v07/data_quality.json
```

Train and evaluate on at least 15 labeled task groups:

```bash
python3 -m foresight_agent.evaluation.external_training \
  data/external/normalized.jsonl
```

Version 0.7 writes a saved model, held-out metrics, quality summary, metadata, and task-group split manifest under `experiments/results/v07/`. The included three-row file is illustrative and intentionally too small for training. Privacy redaction is a safeguard, not a substitute for consent, licensing review, and careful data governance.

## Version 0.8 collection and dual annotation

Collect 40 trajectory steps from 20 curated two-step tasks using real calculator and local-text tool executions:

```bash
python3 -m foresight_agent.data.collector
```

This writes `data/collection/v08/raw_trajectories.jsonl` and a versioned `manifest.json`. No network or API key is used.

Create two independent annotation assignments:

```bash
python3 -m foresight_agent.data.annotation assign \
  data/collection/v08/raw_trajectories.jsonl \
  data/collection/v08/annotator-a.csv --annotator reviewer-a

python3 -m foresight_agent.data.annotation assign \
  data/collection/v08/raw_trajectories.jsonl \
  data/collection/v08/annotator-b.csv --annotator reviewer-b
```

Reviewers should follow `data/collection/ANNOTATION_GUIDE.md` and fill `actual_failure` and `failure_type` independently. Reconcile their completed files:

```bash
python3 -m foresight_agent.data.annotation reconcile \
  data/collection/v08/annotator-a.csv \
  data/collection/v08/annotator-b.csv
```

This produces binary and category Cohen’s kappa, agreed labels, conflicts, and `review/adjudication.csv`. After a lead reviewer completes the final columns in that adjudication file, resolve the dataset:

```bash
python3 -m foresight_agent.data.annotation resolve \
  data/collection/v08/annotator-a.csv \
  data/collection/v08/annotator-b.csv \
  data/collection/v08/review/adjudication.csv \
  data/collection/v08/final_labeled.jsonl
```

Finally train the external predictor:

```bash
python3 -m foresight_agent.evaluation.external_training \
  data/collection/v08/final_labeled.jsonl \
  --output-dir experiments/results/v08
```

The catalog includes successful retrieval/arithmetic tasks and deliberately challenging tool-robustness cases. Human labels—not task names—determine the training targets.

For pipeline development only, generate transparent provisional labels from tool errors and empty searches:

```bash
python3 -m foresight_agent.data.annotation bootstrap \
  data/collection/v08/raw_trajectories.jsonl \
  data/collection/v08/provisional_labeled.jsonl
```

These labels are marked `automated-bootstrap-v0.8`. They can exercise training immediately but must not be reported as independent human annotation or used to calculate inter-annotator agreement.

## Version 0.9 reproducible research package

Run the complete offline development pipeline with one command:

```bash
python3 -m foresight_agent.research.pipeline
```

It collects the curated tasks, creates explicitly marked bootstrap labels, trains external models, audits the dataset and task-group splits, and creates an immutable timestamped package under `experiments/results/v09/`.

For genuine human-reviewed data, provide the resolved dataset:

```bash
python3 -m foresight_agent.research.pipeline \
  --dataset data/collection/v08/final_labeled.jsonl
```

Each packaged run contains:

- A unique run ID and run manifest
- Frozen configuration and Python/dependency environment
- Dataset and model SHA-256 checksums
- Data-quality and split-leakage audit
- Model card and dataset datasheet
- Raw model metrics and split manifest
- Publication-ready SVG metric chart
- Saved calibrated model

The research protocol is documented in `docs/METHODOLOGY.md`. GitHub Actions runs the full test suite using Python 3.12 and the optional ML dependencies.

## Version 1.0 release candidate

Create a report draft from the latest packaged run:

```bash
python3 -m foresight_agent.research.release draft \
  --dataset data/collection/v08/provisional_labeled.jsonl
```

Evaluate final-release readiness after human annotation and adjudication:

```bash
python3 -m foresight_agent.research.release gate \
  --dataset data/collection/v08/final_labeled.jsonl \
  --agreement data/collection/v08/review/agreement.json
```

The gate requires a passing package audit, at least 15 task groups, both label classes, no bootstrap labels, two distinct reviewers, binary Cohen’s kappa of at least 0.60, and a resolved label for every reviewed row. See `docs/RELEASE_CHECKLIST.md`, `docs/ARCHITECTURE.md`, and `docs/METHODOLOGY.md`.

The release candidate must not be renamed `1.0.0` until the gate passes on genuine human-reviewed evidence.
