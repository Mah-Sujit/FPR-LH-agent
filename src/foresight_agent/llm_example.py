from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.llm_planner import StructuredLLMPlanner
from foresight_agent.llm.mock import MockLLMClient
from foresight_agent.tools import CalculatorTool, ToolRegistry


def main() -> None:
    plan = '{"steps":[{"name":"calculate","tool":"calculator","arguments":{"expression":"(12 + 8) * 3"}}]}'
    result = AgentExecutor(StructuredLLMPlanner(MockLLMClient([plan]), ToolRegistry([CalculatorTool()]))).run(
        "Calculate (12 + 8) times 3")
    print(f"Status: {result.status}\nAnswer: {result.state.memory.get('calculate')}")


if __name__ == "__main__":
    main()
