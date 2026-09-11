"""Optional real-model example. Requires OPENAI_API_KEY and the llm extra."""

import argparse
import getpass
import os

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.llm_planner import StructuredLLMPlanner
from foresight_agent.llm.openai_client import OpenAIResponsesClient
from foresight_agent.telemetry import LLMTelemetryLogger
from foresight_agent.tools import CalculatorTool, ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("goal", nargs="?", default="Calculate the percentage increase from 80 to 100")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("OpenAI API key (hidden; paste and press Enter): ").strip()
    if not api_key.startswith("sk-"):
        raise SystemExit("The supplied value does not look like an OpenAI API key.")
    client = OpenAIResponsesClient(
        api_key=api_key,
        telemetry=LLMTelemetryLogger("data/telemetry/openai.jsonl"),
    )
    planner = StructuredLLMPlanner(client, ToolRegistry([CalculatorTool()]))
    result = AgentExecutor(planner).run(args.goal)
    print(f"Status: {result.status}")
    print(f"Memory: {result.state.memory}")


if __name__ == "__main__":
    main()
