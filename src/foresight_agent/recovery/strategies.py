"""State transitions that implement recovery decisions."""

from __future__ import annotations

from copy import deepcopy

from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState

from .controller import RecoveryAction


class RecoveryStrategies:
    def apply(self, action: RecoveryAction, state: AgentState, planner: Planner) -> None:
        if action is RecoveryAction.CONTINUE:
            state.snapshots.append((state.current_step, deepcopy(state.memory)))
            state.current_step += 1
        elif action is RecoveryAction.VERIFY:
            state.verified_steps.add(state.current_step)
        elif action is RecoveryAction.RETRY:
            state.retries[state.current_step] = state.retries.get(state.current_step, 0) + 1
        elif action is RecoveryAction.REPLAN:
            state.plan = planner.replan(state)
            state.replans += 1
        elif action is RecoveryAction.ROLLBACK:
            step, memory = state.snapshots.pop()
            state.memory = memory
            state.current_step = step
            state.rollbacks += 1
        # ABSTAIN is terminal and needs no state mutation.

