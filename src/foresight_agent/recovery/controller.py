"""Map risk and execution context to a recovery action."""

from __future__ import annotations

from enum import Enum

from foresight_agent.agent.state import AgentState, StepRecord


class RecoveryAction(str, Enum):
    CONTINUE = "CONTINUE"
    VERIFY = "VERIFY"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    ROLLBACK = "ROLLBACK"
    ABSTAIN = "ABSTAIN"


class RecoveryController:
    """A conservative, deterministic recovery policy."""

    def __init__(
        self,
        max_retries: int = 2,
        max_replans: int = 1,
        max_rollbacks: int = 1,
        verify_threshold: float = 0.30,
        replan_threshold: float = 0.75,
    ) -> None:
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.max_rollbacks = max_rollbacks
        self.verify_threshold = verify_threshold
        self.replan_threshold = replan_threshold

    def decide(self, state: AgentState, record: StepRecord) -> RecoveryAction:
        risk = record.risk_score
        retries = state.retries.get(state.current_step, 0)

        if record.signals.get("contradiction", 0) and state.snapshots and state.rollbacks < self.max_rollbacks:
            return RecoveryAction.ROLLBACK
        if not record.success and retries < self.max_retries:
            return RecoveryAction.RETRY
        if risk >= self.replan_threshold and state.replans < self.max_replans:
            return RecoveryAction.REPLAN
        if risk >= self.replan_threshold:
            return RecoveryAction.ABSTAIN
        if risk >= self.verify_threshold and state.current_step not in state.verified_steps:
            return RecoveryAction.VERIFY
        return RecoveryAction.CONTINUE
