"""Minimal planner abstraction for Version 0.1."""

from __future__ import annotations

from collections.abc import Callable

from .state import AgentState, TaskStep


class Planner:
    """Creates a plan and optionally repairs its remaining portion.

    A future LLM planner can implement the same two methods.
    """

    def __init__(
        self,
        steps: list[TaskStep],
        replan_hook: Callable[[AgentState], list[TaskStep]] | None = None,
    ) -> None:
        self._steps = steps
        self._replan_hook = replan_hook

    def create_plan(self, goal: str) -> list[TaskStep]:
        del goal
        return list(self._steps)

    def replan(self, state: AgentState) -> list[TaskStep]:
        if self._replan_hook is None:
            return state.plan
        return self._replan_hook(state)

