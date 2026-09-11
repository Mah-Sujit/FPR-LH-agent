"""Runnable, API-key-free demonstration of retry and verification."""

from __future__ import annotations

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep
from foresight_agent.monitoring.trajectory_logger import TrajectoryLogger


def collect_sources(state: AgentState) -> dict[str, object]:
    state.memory["sources"] = ["paper-a", "paper-b", "paper-c"]
    return {"observation": "Collected three sources.", "progress_score": 0.30, "confidence": 0.95}


def extract_evidence(state: AgentState) -> dict[str, object]:
    attempts = state.retries.get(state.current_step, 0)
    if attempts == 0:
        return {
            "observation": "The document parser timed out.",
            "success": False,
            "confidence": 0.20,
            "progress_score": 0.30,
            "error": "temporary parser timeout",
        }
    state.memory["evidence"] = ["finding-1", "finding-2"]
    return {"observation": "Evidence extracted after retry.", "progress_score": 0.65, "confidence": 0.90}


def write_summary(state: AgentState) -> dict[str, object]:
    state.memory["summary"] = "Three sources support two main findings."
    return {"observation": state.memory["summary"], "progress_score": 1.0, "confidence": 0.85}


def main() -> None:
    steps = [
        TaskStep("collect_sources", collect_sources, "Gather relevant sources"),
        TaskStep("extract_evidence", extract_evidence, "Extract useful evidence"),
        TaskStep("write_summary", write_summary, "Synthesize the result"),
    ]
    executor = AgentExecutor(
        planner=Planner(steps),
        logger=TrajectoryLogger("data/trajectories/example.jsonl"),
    )
    result = executor.run("Research a topic and produce an evidence-based summary")

    print(f"Status: {result.status}")
    for record in result.state.trajectory:
        print(
            f"Step {record.step}: {record.action:<18} "
            f"risk={record.risk_score:.3f} recovery={record.recovery_action}"
        )
    print(f"Result: {result.state.memory.get('summary', 'No summary produced')}")


if __name__ == "__main__":
    main()

