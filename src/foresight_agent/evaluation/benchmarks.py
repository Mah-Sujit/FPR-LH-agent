"""Small deterministic benchmark suite for Version 0.2."""

from dataclasses import dataclass
from typing import Callable

from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep


@dataclass
class BenchmarkCase:
    name: str
    failure_type: str
    build_planner: Callable[[], Planner]
    success_check: Callable[[AgentState], bool]


def _healthy() -> Planner:
    def execute(state: AgentState):
        state.memory["answer"] = "correct"
        return {"observation": "Task completed", "confidence": 0.98, "progress_score": 1.0}
    return Planner([TaskStep("healthy_action", execute)])


def _transient_tool_failure() -> Planner:
    def execute(state: AgentState):
        if state.retries.get(state.current_step, 0) == 0:
            return {"observation": "Temporary timeout", "success": False, "confidence": 0.1,
                    "progress_score": 0.0, "actual_failure": True, "failure_type": "tool_failure"}
        state.memory["answer"] = "recovered"
        return {"observation": "Retry succeeded", "confidence": 0.95, "progress_score": 1.0}
    return Planner([TaskStep("flaky_tool", execute)])


def _low_confidence() -> Planner:
    def execute(state: AgentState):
        if state.current_step not in state.verified_steps:
            return {"observation": "Ambiguous evidence repeated", "confidence": 0.2, "progress_score": 0.4,
                    "repeated_action": True, "actual_failure": True, "failure_type": "missing_information"}
        state.memory["answer"] = "verified"
        return {"observation": "Evidence verified", "confidence": 0.95, "progress_score": 1.0}
    return Planner([TaskStep("inspect_evidence", execute)])


def _persistent_failure() -> Planner:
    def execute(state: AgentState):
        del state
        return {"observation": "Service unavailable", "success": False, "confidence": 0.0,
                "progress_score": 0.0, "actual_failure": True, "failure_type": "tool_failure"}
    return Planner([TaskStep("unavailable_tool", execute)])


def default_benchmarks() -> list[BenchmarkCase]:
    def has_answer(state: AgentState) -> bool:
        return "answer" in state.memory
    return [
        BenchmarkCase("healthy", "none", _healthy, has_answer),
        BenchmarkCase("transient_tool_failure", "tool_failure", _transient_tool_failure, has_answer),
        BenchmarkCase("low_confidence", "missing_information", _low_confidence, has_answer),
        BenchmarkCase("persistent_failure", "tool_failure", _persistent_failure, lambda state: False),
    ]
