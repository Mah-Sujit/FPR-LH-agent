import json
from typing import Any

from foresight_agent.llm.base import LLMClient
from foresight_agent.tools.base import ToolRegistry
from .planner import Planner
from .state import AgentState, TaskStep


class StructuredLLMPlanner(Planner):
    SYSTEM_PROMPT = ('Return JSON only: {"steps":[{"name":"short_name","tool":"tool_name",'
                     '"arguments":{}}]}. Use only listed tools and create 1 to 10 steps.')

    def __init__(self, client: LLMClient, tools: ToolRegistry) -> None:
        super().__init__([])
        self.client, self.tools = client, tools

    def create_plan(self, goal: str) -> list[TaskStep]:
        prompt = json.dumps({"goal": goal, "tools": self.tools.descriptions()}, sort_keys=True)
        return self._parse_plan(self.client.generate(self.SYSTEM_PROMPT, prompt).text)

    def replan(self, state: AgentState) -> list[TaskStep]:
        prompt = json.dumps({"goal": state.goal, "tools": self.tools.descriptions(),
                             "previous_actions": [r.action for r in state.trajectory],
                             "instruction": "Produce a safer replacement plan."}, sort_keys=True)
        return self._parse_plan(self.client.generate(self.SYSTEM_PROMPT, prompt).text)

    def _parse_plan(self, text: str) -> list[TaskStep]:
        try:
            raw_steps = json.loads(text)["steps"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("Planner response is not valid plan JSON.") from exc
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 10:
            raise ValueError("A plan must contain between 1 and 10 steps.")
        return [self._make_step(raw, index) for index, raw in enumerate(raw_steps, 1)]

    def _make_step(self, raw: Any, index: int) -> TaskStep:
        if not isinstance(raw, dict) or not isinstance(raw.get("arguments", {}), dict):
            raise ValueError(f"Plan step {index} is malformed.")
        name, tool_name = str(raw.get("name", "")).strip(), str(raw.get("tool", "")).strip()
        if not name or not tool_name:
            raise ValueError(f"Plan step {index} requires name and tool.")
        tool, arguments = self.tools.get(tool_name), dict(raw.get("arguments", {}))

        def execute(state: AgentState, selected=tool, supplied=arguments, step_name=name):
            result = selected.run(supplied)
            state.memory[step_name] = result.get("observation")
            progress = min(1.0, (state.current_step + 1) / max(1, len(state.plan)))
            return {"progress_score": progress, **result}
        return TaskStep(name, execute, f"Use {tool_name}")
