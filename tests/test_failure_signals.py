from foresight_agent.agent.state import AgentState, StepRecord
from foresight_agent.monitoring.failure_signals import FailureSignalDetector


def test_detects_error_low_confidence_and_stalled_progress():
    state = AgentState(goal="test", plan=[])
    state.trajectory.append(StepRecord(1, "test", "search", "ok", True, progress_score=0.5))
    record = StepRecord(2, "test", "search", "failed", False, confidence=0.2, progress_score=0.5)

    signals = FailureSignalDetector().detect(state, record)

    assert signals["tool_error"] == 1.0
    assert signals["low_confidence"] == 0.8
    assert signals["no_progress"] == 1.0

