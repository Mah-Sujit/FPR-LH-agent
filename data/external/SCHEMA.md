# External trajectory schema

Each JSONL object or CSV row represents one agent step.

Required fields: `dataset_id`, `task_id`, `run_id`, `step`, `goal`, `action`, `observation`, `success`, `confidence`, and `progress_score`.

Human labels: `actual_failure` must be `true` or `false`. A true label requires one of: `tool_execution_error`, `tool_selection_error`, `planning_error`, `repetition_loop`, `contradiction`, `hallucination`, `missing_evidence`, or `other`. A false label uses `none` or an empty value.

Optional execution fields: `repeated_action`, `contradiction`, `retry_count`, `step_position`, `progress_delta`, `previous_risk`, `annotator_id`, and `source`.

All probability-like fields must fall in `[0, 1]`; `progress_delta` must fall in `[-1, 1]`. The tuple `(dataset_id, task_id, run_id, step)` must be unique. Observations are limited to 20,000 characters during normalization. API-key-like strings and email addresses are redacted, but collectors remain responsible for consent, licensing, and removal of personal or confidential data.
