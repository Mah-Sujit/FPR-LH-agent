# Annotation guide

Judge each step using the task goal, action, and observation. Label `actual_failure=True` when the step is incorrect, unusable, unsupported by evidence, or prevents successful completion—even if the tool reported success. Otherwise label it `False` with `failure_type=none`.

Failure categories:

- `tool_execution_error`: the selected tool failed to execute correctly.
- `tool_selection_error`: the action used an unsuitable tool.
- `planning_error`: the step ordering or decomposition is wrong.
- `repetition_loop`: the agent repeats work without useful progress.
- `contradiction`: the result conflicts with established trajectory information.
- `hallucination`: the result asserts unsupported or fabricated information.
- `missing_evidence`: necessary evidence was not found or checked.
- `other`: a genuine failure not covered above; explain it in adjudication notes.

Annotators must work independently. Do not inspect the other annotator’s file before submission. During adjudication, review only rows listed in `adjudication.csv`, enter `final_actual_failure`, `final_failure_type`, `adjudicator_id`, and optional notes. Do not change identifiers or source observations.
