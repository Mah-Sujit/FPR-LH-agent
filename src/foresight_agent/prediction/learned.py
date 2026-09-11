"""Persisted learned risk scorer compatible with AgentExecutor."""

from __future__ import annotations

from pathlib import Path

from foresight_agent.agent.state import AgentState, StepRecord
from .features import feature_vector


class LearnedRiskScorer:
    def __init__(self, model_or_path) -> None:
        if isinstance(model_or_path, (str, Path)):
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc
            self.model = joblib.load(model_or_path)
        else:
            self.model = model_or_path

    def score_record(self, state: AgentState, record: StepRecord) -> float:
        del state
        probability = self.model.predict_proba([feature_vector(record)])[0][1]
        return round(float(probability), 6)

