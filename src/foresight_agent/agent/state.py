"""Shared, serializable state models used by every module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


StepFunction = Callable[["AgentState"], Any]


@dataclass
class TaskStep:
    """One named unit of work in an agent plan."""

    name: str
    action: StepFunction
    description: str = ""


@dataclass
class StepRecord:
    """A single trajectory event and its monitoring metadata."""

    step: int
    goal: str
    action: str
    observation: str
    success: bool
    confidence: float = 1.0
    progress_score: float = 0.0
    repeated_action: bool = False
    contradiction: bool = False
    error: str | None = None
    actual_failure: bool = False
    failure_type: str | None = None
    retry_count: int = 0
    step_position: float = 0.0
    progress_delta: float = 0.0
    previous_risk: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    recovery_action: str = "CONTINUE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Mutable execution state owned by one agent run."""

    goal: str
    plan: list[TaskStep]
    current_step: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    trajectory: list[StepRecord] = field(default_factory=list)
    retries: dict[int, int] = field(default_factory=dict)
    replans: int = 0
    rollbacks: int = 0
    verified_steps: set[int] = field(default_factory=set)
    snapshots: list[tuple[int, dict[str, Any]]] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.current_step >= len(self.plan)
