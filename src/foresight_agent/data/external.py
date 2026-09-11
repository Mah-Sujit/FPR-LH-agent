"""Strict JSONL/CSV interchange format for externally collected trajectories."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from foresight_agent.agent.state import StepRecord
from foresight_agent.monitoring.failure_signals import FailureSignalDetector
from foresight_agent.prediction.risk_score import RiskScorer


FAILURE_TYPES = {
    "none", "tool_execution_error", "tool_selection_error", "planning_error",
    "repetition_loop", "contradiction", "hallucination", "missing_evidence", "other",
}
_SECRET = re.compile(r"(?i)(?:bearer\s+)?sk-[A-Za-z0-9_-]{12,}")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class DatasetValidationError(ValueError):
    pass


def redact_text(value: str) -> str:
    return _EMAIL.sub("[REDACTED_EMAIL]", _SECRET.sub("[REDACTED_SECRET]", value))[:20_000]


def contains_sensitive_text(value: str) -> bool:
    """Detect only key-shaped secrets or email addresses, avoiding `task-` false positives."""
    return bool(_SECRET.search(value) or _EMAIL.search(value))


@dataclass
class ExternalTrajectory:
    dataset_id: str
    task_id: str
    run_id: str
    step: int
    goal: str
    action: str
    observation: str
    success: bool
    confidence: float
    progress_score: float
    repeated_action: bool = False
    contradiction: bool = False
    retry_count: int = 0
    step_position: float = 0.0
    progress_delta: float = 0.0
    previous_risk: float = 0.0
    actual_failure: bool | None = None
    failure_type: str | None = None
    annotator_id: str | None = None
    source: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], location: str) -> "ExternalTrajectory":
        required = {"dataset_id", "task_id", "run_id", "step", "goal", "action", "observation",
                    "success", "confidence", "progress_score"}
        missing = sorted(required - set(raw))
        if missing:
            raise DatasetValidationError(f"{location}: missing fields {missing}")
        try:
            item = cls(
                dataset_id=str(raw["dataset_id"]).strip(), task_id=str(raw["task_id"]).strip(),
                run_id=str(raw["run_id"]).strip(), step=int(raw["step"]), goal=redact_text(str(raw["goal"])),
                action=str(raw["action"]).strip(), observation=redact_text(str(raw["observation"])),
                success=_boolean(raw["success"], "success"), confidence=float(raw["confidence"]),
                progress_score=float(raw["progress_score"]),
                repeated_action=_boolean(raw.get("repeated_action", False), "repeated_action"),
                contradiction=_boolean(raw.get("contradiction", False), "contradiction"),
                retry_count=int(raw.get("retry_count", 0)), step_position=float(raw.get("step_position", 0.0)),
                progress_delta=float(raw.get("progress_delta", 0.0)),
                previous_risk=float(raw.get("previous_risk", 0.0)),
                actual_failure=_optional_boolean(raw.get("actual_failure")),
                failure_type=_optional(raw.get("failure_type")), annotator_id=_optional(raw.get("annotator_id")),
                source=_optional(raw.get("source")),
            )
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(f"{location}: {exc}") from exc
        item.validate(location)
        return item

    def validate(self, location: str = "row") -> None:
        if not all((self.dataset_id, self.task_id, self.run_id, self.action)):
            raise DatasetValidationError(f"{location}: identifiers and action cannot be empty")
        if self.step < 1 or self.retry_count < 0:
            raise DatasetValidationError(f"{location}: step must be >= 1 and retry_count >= 0")
        for name in ("confidence", "progress_score", "step_position", "previous_risk"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise DatasetValidationError(f"{location}: {name} must be in [0, 1]")
        if not -1.0 <= self.progress_delta <= 1.0:
            raise DatasetValidationError(f"{location}: progress_delta must be in [-1, 1]")
        if self.failure_type is not None and self.failure_type not in FAILURE_TYPES:
            raise DatasetValidationError(f"{location}: unsupported failure_type {self.failure_type!r}")
        if self.actual_failure is False and self.failure_type not in (None, "none"):
            raise DatasetValidationError(f"{location}: non-failure cannot have a failure category")
        if self.actual_failure is True and self.failure_type in (None, "none"):
            raise DatasetValidationError(f"{location}: failure label requires a failure_type")

    def to_record(self) -> StepRecord:
        record = StepRecord(self.step, self.goal, self.action, self.observation, self.success,
                            self.confidence, self.progress_score, self.repeated_action, self.contradiction,
                            actual_failure=bool(self.actual_failure), failure_type=self.failure_type,
                            retry_count=self.retry_count, step_position=self.step_position,
                            progress_delta=self.progress_delta, previous_risk=self.previous_risk)
        state = type("ExternalState", (), {"trajectory": []})()
        record.signals = FailureSignalDetector().detect(state, record)
        record.risk_score = RiskScorer().score(record.signals)
        return record


def _optional(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be a boolean")


def _optional_boolean(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return _boolean(value, "actual_failure")


def load_trajectories(path: str | Path, require_labels: bool = False) -> list[ExternalTrajectory]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        raw_rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    raw_rows.append((f"line {line_number}", json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(f"line {line_number}: invalid JSON") from exc
    elif source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as stream:
            raw_rows = [(f"row {index}", row) for index, row in enumerate(csv.DictReader(stream), 2)]
    else:
        raise DatasetValidationError("Input must be .jsonl or .csv")
    items = [ExternalTrajectory.from_mapping(raw, location) for location, raw in raw_rows]
    if not items:
        raise DatasetValidationError("Dataset is empty")
    keys = [(item.dataset_id, item.task_id, item.run_id, item.step) for item in items]
    if len(keys) != len(set(keys)):
        raise DatasetValidationError("Dataset contains duplicate dataset/task/run/step keys")
    if require_labels:
        unlabeled = [f"{item.task_id}/{item.run_id}/step-{item.step}"
                     for item in items if item.actual_failure is None]
        if unlabeled:
            preview = ", ".join(unlabeled[:10])
            suffix = f" (and {len(unlabeled) - 10} more)" if len(unlabeled) > 10 else ""
            raise DatasetValidationError(
                f"{len(unlabeled)} row(s) still need actual_failure and failure_type labels: "
                f"{preview}{suffix}"
            )
    return items


def write_normalized_jsonl(items: list[ExternalTrajectory], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(asdict(item), sort_keys=True) + "\n")
