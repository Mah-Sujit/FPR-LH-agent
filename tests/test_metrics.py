from foresight_agent.agent.state import StepRecord
from foresight_agent.evaluation.metrics import prediction_metrics


def record(risk, actual, action="CONTINUE"):
    item = StepRecord(1, "goal", "action", "observation", not actual)
    item.risk_score = risk
    item.actual_failure = actual
    item.recovery_action = action
    return item


def test_prediction_metrics():
    metrics = prediction_metrics([
        record(0.8, True, "RETRY"),
        record(0.7, False, "VERIFY"),
        record(0.1, False),
        record(0.1, True),
    ], threshold=0.3)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.accuracy == 0.5
    assert metrics.unnecessary_intervention_rate == 0.5

