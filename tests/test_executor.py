from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep


def test_executor_retries_failure_then_completes():
    def flaky(state: AgentState):
        if state.retries.get(state.current_step, 0) == 0:
            return {"success": False, "confidence": 0.1, "progress_score": 0.0}
        state.memory["done"] = True
        return {"success": True, "confidence": 1.0, "progress_score": 1.0}

    result = AgentExecutor(Planner([TaskStep("flaky", flaky)])).run("complete task")

    assert result.status == "completed"
    assert result.state.memory["done"] is True
    assert [r.recovery_action for r in result.state.trajectory] == ["RETRY", "CONTINUE"]


def test_executor_abstains_at_iteration_guard():
    def uncertain(state: AgentState):
        return {"success": True, "confidence": 0.6, "progress_score": 0.0}

    result = AgentExecutor(Planner([TaskStep("uncertain", uncertain)]), max_iterations=1).run("task")
    assert result.status == "abstained"

