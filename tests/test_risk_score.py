from foresight_agent.prediction.risk_score import RiskScorer


def test_score_is_weighted_and_bounded():
    scorer = RiskScorer()
    assert scorer.score({"tool_error": 1.0, "low_confidence": 1.0}) == 0.65
    assert 0.0 <= scorer.score({"tool_error": 99.0}) <= 1.0

