"""Shared construction options for recovery-policy experiments."""

from foresight_agent.config import ExperimentConfig


def controller_options(config: ExperimentConfig, threshold: float | None = None) -> dict[str, int | float]:
    return {"max_retries": config.max_retries, "max_replans": config.max_replans,
            "max_rollbacks": config.max_rollbacks,
            "verify_threshold": config.prediction_threshold if threshold is None else threshold,
            "replan_threshold": config.replan_threshold}
