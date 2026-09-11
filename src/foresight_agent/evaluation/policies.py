"""Controller baselines used in comparative experiments."""

from foresight_agent.agent.state import AgentState, StepRecord
from foresight_agent.recovery.controller import RecoveryAction, RecoveryController


class NoRecoveryController(RecoveryController):
    """Ablation baseline that accepts every step, including failed ones."""

    def decide(self, state: AgentState, record: StepRecord) -> RecoveryAction:
        del state, record
        return RecoveryAction.CONTINUE


class OracleRecoveryController(RecoveryController):
    """Upper-bound policy allowed to see benchmark ground-truth labels."""

    def decide(self, state: AgentState, record: StepRecord) -> RecoveryAction:
        if not record.actual_failure:
            return RecoveryAction.CONTINUE
        if record.failure_type in {"missing_information", "missing_evidence"} and state.current_step not in state.verified_steps:
            return RecoveryAction.VERIFY
        if record.failure_type == "contradiction" and state.snapshots and state.rollbacks < self.max_rollbacks:
            return RecoveryAction.ROLLBACK
        if state.retries.get(state.current_step, 0) < self.max_retries:
            return RecoveryAction.RETRY
        if state.replans < self.max_replans:
            return RecoveryAction.REPLAN
        return RecoveryAction.ABSTAIN
