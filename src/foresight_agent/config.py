"""JSON-backed experiment configuration with safe defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from foresight_agent.prediction.risk_score import RiskScorer


@dataclass
class ExperimentConfig:
    predictor: str = "rule_based"
    model_path: str = "experiments/results/v06/best_model.joblib"
    prediction_threshold: float = 0.30
    replan_threshold: float = 0.75
    max_retries: int = 2
    max_replans: int = 1
    max_rollbacks: int = 1
    max_iterations: int = 50
    runs_per_task: int = 3
    seed: int = 42
    noisy_runs_per_task: int = 10
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    risk_weights: dict[str, float] = field(default_factory=lambda: dict(RiskScorer.DEFAULT_WEIGHTS))

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values: dict[str, Any] = json.load(stream)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
