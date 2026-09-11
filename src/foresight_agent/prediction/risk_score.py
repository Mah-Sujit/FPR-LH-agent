"""Explainable weighted risk score for Version 0.1."""

from __future__ import annotations


class RiskScorer:
    DEFAULT_WEIGHTS = {
        "tool_error": 0.45,
        "low_confidence": 0.20,
        "repetition": 0.15,
        "contradiction": 0.15,
        "no_progress": 0.05,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def score(self, signals: dict[str, float]) -> float:
        total_weight = sum(self.weights.values()) or 1.0
        raw = sum(self.weights.get(name, 0.0) * value for name, value in signals.items())
        return round(max(0.0, min(1.0, raw / total_weight)), 3)

