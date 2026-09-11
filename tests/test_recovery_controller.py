from foresight_agent.agent.state import AgentState, StepRecord
from foresight_agent.recovery.controller import RecoveryAction, RecoveryController


def make_record(success=True, risk=0.0, **signals):
    record = StepRecord(1, "goal", "action", "observation", success)
    record.risk_score = risk
    record.signals = signals
    return record


def test_controller_covers_all_decisions():
    controller = RecoveryController(max_retries=1, max_replans=1, max_rollbacks=1)
    state = AgentState("goal", [])

    assert controller.decide(state, make_record(risk=0.1)) is RecoveryAction.CONTINUE
    assert controller.decide(state, make_record(risk=0.4)) is RecoveryAction.VERIFY
    assert controller.decide(state, make_record(success=False, risk=0.7)) is RecoveryAction.RETRY

    state.retries[0] = 1
    assert controller.decide(state, make_record(success=False, risk=0.8)) is RecoveryAction.REPLAN
    state.replans = 1
    assert controller.decide(state, make_record(success=False, risk=0.8)) is RecoveryAction.ABSTAIN

    state.snapshots.append((0, {}))
    assert controller.decide(state, make_record(risk=0.6, contradiction=1.0)) is RecoveryAction.ROLLBACK

