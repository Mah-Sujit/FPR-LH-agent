"""A 24-task offline suite spanning common long-horizon failure modes."""

from __future__ import annotations

from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep

from .benchmarks import BenchmarkCase


def _success_step(label: str):
    def execute(state: AgentState):
        state.memory["answer"] = label
        return {"observation": label, "confidence": 0.96, "progress_score": 1.0}
    return execute


def _healthy(label: str) -> Planner:
    return Planner([TaskStep("complete", _success_step(label))])


def _retryable(label: str, failure_type: str, repeated: bool = False) -> Planner:
    def execute(state: AgentState):
        if state.retries.get(state.current_step, 0) == 0:
            return {"observation": f"Injected {failure_type}", "success": False, "confidence": 0.1,
                    "progress_score": 0.0, "repeated_action": repeated,
                    "actual_failure": True, "failure_type": failure_type}
        return _success_step(label)(state)
    return Planner([TaskStep("recoverable_action", execute)])


def _missing_evidence(label: str) -> Planner:
    def execute(state: AgentState):
        if state.current_step not in state.verified_steps:
            return {"observation": "Evidence is incomplete", "confidence": 0.15, "progress_score": 0.2,
                    "repeated_action": True, "actual_failure": True, "failure_type": "missing_evidence"}
        return _success_step(label)(state)
    return Planner([TaskStep("verify_evidence", execute)])


def _replan_case(label: str, failure_type: str) -> Planner:
    def bad_step(state: AgentState):
        del state
        return {"observation": f"Invalid plan: {failure_type}", "success": False, "confidence": 0.05,
                "progress_score": 0.0, "actual_failure": True, "failure_type": failure_type}

    def replan(state: AgentState):
        del state
        return [TaskStep("replacement_step", _success_step(label))]
    return Planner([TaskStep("bad_step", bad_step)], replan_hook=replan)


def _rollback_case(label: str, failure_type: str) -> Planner:
    attempts = {"count": 0}

    def prepare(state: AgentState):
        state.memory["prepared"] = True
        return {"observation": "Checkpoint prepared", "confidence": 0.98, "progress_score": 0.4}

    def inspect(state: AgentState):
        attempts["count"] += 1
        if attempts["count"] == 1:
            state.memory["corrupted"] = True
            return {"observation": f"Detected {failure_type}", "confidence": 0.1, "progress_score": 0.3,
                    "contradiction": True, "actual_failure": True, "failure_type": failure_type}
        return _success_step(label)(state)
    return Planner([TaskStep("prepare", prepare), TaskStep("inspect", inspect)])


def offline_benchmarks(variants: int = 3) -> list[BenchmarkCase]:
    """Return 24 cases by default: three variants of eight categories."""
    if variants < 1:
        raise ValueError("variants must be at least 1")
    cases: list[BenchmarkCase] = []
    has_answer = lambda state: "answer" in state.memory
    for number in range(1, variants + 1):
        label = f"answer-{number}"
        builders = [
            ("healthy", "none", lambda value=label: _healthy(value)),
            ("tool_execution", "tool_execution_error", lambda value=label: _retryable(value, "tool_execution_error")),
            ("repetition", "repetition_loop", lambda value=label: _retryable(value, "repetition_loop", True)),
            ("missing_evidence", "missing_evidence", lambda value=label: _missing_evidence(value)),
            ("planning", "planning_error", lambda value=label: _replan_case(value, "planning_error")),
            ("tool_selection", "tool_selection_error", lambda value=label: _replan_case(value, "tool_selection_error")),
            ("contradiction", "contradiction", lambda value=label: _rollback_case(value, "contradiction")),
            ("hallucination", "hallucination", lambda value=label: _rollback_case(value, "hallucination")),
        ]
        cases.extend(BenchmarkCase(f"{name}_{number:02d}", kind, builder, has_answer)
                     for name, kind, builder in builders)
    return cases

