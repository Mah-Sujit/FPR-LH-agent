import pytest

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.llm_planner import StructuredLLMPlanner
from foresight_agent.llm.mock import MockLLMClient
from foresight_agent.tools import CalculatorTool, ToolRegistry


def test_mock_llm_plan_executes_tool():
    plan = '{"steps":[{"name":"total","tool":"calculator","arguments":{"expression":"6 * 7"}}]}'
    client = MockLLMClient([plan])
    result = AgentExecutor(StructuredLLMPlanner(client, ToolRegistry([CalculatorTool()]))).run("find total")
    assert result.status == "completed"
    assert result.state.memory["total"] == "42"
    assert len(client.calls) == 1


def test_planner_rejects_unknown_tool():
    plan = '{"steps":[{"name":"bad","tool":"shell","arguments":{}}]}'
    planner = StructuredLLMPlanner(MockLLMClient([plan]), ToolRegistry([CalculatorTool()]))
    with pytest.raises(ValueError, match="Unknown tool"):
        planner.create_plan("unsafe task")


def test_planner_rejects_invalid_json():
    planner = StructuredLLMPlanner(MockLLMClient(["not json"]), ToolRegistry([CalculatorTool()]))
    with pytest.raises(ValueError, match="valid plan JSON"):
        planner.create_plan("task")
