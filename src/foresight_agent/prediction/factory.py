"""Create a configured rule-based or learned risk scorer."""

from foresight_agent.config import ExperimentConfig
from .learned import LearnedRiskScorer
from .risk_score import RiskScorer


def create_scorer(config: ExperimentConfig):
    if config.predictor == "rule_based":
        return RiskScorer(config.risk_weights)
    if config.predictor in {"learned", "logistic_regression", "decision_tree", "random_forest"}:
        return LearnedRiskScorer(config.model_path)
    raise ValueError(f"Unknown predictor: {config.predictor}")
