"""Dependency-free metrics for failure prediction and recovery."""

from __future__ import annotations

from dataclasses import dataclass

from foresight_agent.agent.state import StepRecord


@dataclass
class PredictionMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    warning_lead_time: float
    unnecessary_intervention_rate: float


def prediction_metrics(records: list[StepRecord], threshold: float = 0.30) -> PredictionMetrics:
    predicted = [record.risk_score >= threshold for record in records]
    actual = [record.actual_failure for record in records]
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0
    interventions = [record for record in records if record.recovery_action != "CONTINUE"]
    unnecessary = sum(not record.actual_failure for record in interventions)
    unnecessary_rate = unnecessary / len(interventions) if interventions else 0.0
    leads: list[int] = []
    for index, is_failure in enumerate(actual):
        if is_failure:
            warnings = [i for i in range(index + 1) if predicted[i]]
            if warnings:
                leads.append(index - warnings[0])
    warning_lead = sum(leads) / len(leads) if leads else 0.0
    return PredictionMetrics(precision, recall, f1, accuracy, warning_lead, unnecessary_rate)

