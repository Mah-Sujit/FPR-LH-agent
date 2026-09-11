"""Stable numerical feature schema shared by training and inference."""

from foresight_agent.agent.state import StepRecord


FEATURE_NAMES = (
    "confidence", "tool_success", "progress_score", "progress_delta",
    "repeated_action", "contradiction", "retry_count", "step_position",
    "previous_risk", "signal_tool_error", "signal_low_confidence",
    "signal_repetition", "signal_contradiction", "signal_no_progress",
)


def feature_vector(record: StepRecord) -> list[float]:
    return [
        record.confidence,
        float(record.success),
        record.progress_score,
        record.progress_delta,
        float(record.repeated_action),
        float(record.contradiction),
        float(record.retry_count),
        record.step_position,
        record.previous_risk,
        record.signals.get("tool_error", 0.0),
        record.signals.get("low_confidence", 0.0),
        record.signals.get("repetition", 0.0),
        record.signals.get("contradiction", 0.0),
        record.signals.get("no_progress", 0.0),
    ]
