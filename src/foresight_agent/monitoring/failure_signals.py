"""Transparent rule-based indicators of likely trajectory failure."""

from __future__ import annotations

from foresight_agent.agent.state import AgentState, StepRecord


class FailureSignalDetector:
    """Returns normalized signals in the range [0, 1]."""

    def detect(self, state: AgentState, record: StepRecord) -> dict[str, float]:
        same_action_count = sum(r.action == record.action for r in state.trajectory)
        previous_progress = state.trajectory[-1].progress_score if state.trajectory else 0.0
        stalled = bool(state.trajectory) and record.progress_score <= previous_progress

        return {
            "tool_error": 0.0 if record.success else 1.0,
            "low_confidence": max(0.0, min(1.0, 1.0 - record.confidence)),
            "repetition": 1.0 if record.repeated_action or same_action_count >= 2 else 0.0,
            "contradiction": 1.0 if record.contradiction else 0.0,
            "no_progress": 1.0 if stalled else 0.0,
        }

