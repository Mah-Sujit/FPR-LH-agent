"""The monitored single-agent long-horizon execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from foresight_agent.monitoring.failure_signals import FailureSignalDetector
from foresight_agent.monitoring.trajectory_logger import TrajectoryLogger
from foresight_agent.prediction.risk_score import RiskScorer
from foresight_agent.recovery.controller import RecoveryAction, RecoveryController
from foresight_agent.recovery.strategies import RecoveryStrategies

from .planner import Planner
from .state import AgentState, StepRecord


@dataclass
class ExecutionResult:
    status: str
    state: AgentState
    message: str


class AgentExecutor:
    """Executes plan steps while predicting and recovering from failure."""

    def __init__(
        self,
        planner: Planner,
        logger: TrajectoryLogger | None = None,
        detector: FailureSignalDetector | None = None,
        scorer: RiskScorer | None = None,
        controller: RecoveryController | None = None,
        max_iterations: int = 50,
    ) -> None:
        self.planner = planner
        self.logger = logger or TrajectoryLogger()
        self.detector = detector or FailureSignalDetector()
        self.scorer = scorer or RiskScorer()
        self.controller = controller or RecoveryController()
        self.strategies = RecoveryStrategies()
        self.max_iterations = max_iterations

    def run(self, goal: str) -> ExecutionResult:
        state = AgentState(goal=goal, plan=self.planner.create_plan(goal))

        for iteration in range(self.max_iterations):
            if state.completed:
                return ExecutionResult("completed", state, "Goal completed successfully.")

            task = state.plan[state.current_step]
            record = self._execute_step(state, task.name, task.action, iteration + 1)
            record.retry_count = state.retries.get(state.current_step, 0)
            record.step_position = state.current_step / max(1, len(state.plan) - 1)
            previous_progress = state.trajectory[-1].progress_score if state.trajectory else 0.0
            record.progress_delta = record.progress_score - previous_progress
            record.previous_risk = state.trajectory[-1].risk_score if state.trajectory else 0.0
            record.signals = self.detector.detect(state, record)
            if hasattr(self.scorer, "score_record"):
                record.risk_score = self.scorer.score_record(state, record)
            else:
                record.risk_score = self.scorer.score(record.signals)
            decision = self.controller.decide(state, record)
            record.recovery_action = decision.value
            self.logger.log(state, record)

            if decision is RecoveryAction.ABSTAIN:
                return ExecutionResult("abstained", state, "Risk remained too high; human help is required.")
            self.strategies.apply(decision, state, self.planner)

        return ExecutionResult("abstained", state, "Maximum execution iterations reached.")

    @staticmethod
    def _execute_step(state: AgentState, action_name: str, action: Any, step_number: int) -> StepRecord:
        try:
            result = action(state)
            payload = result if isinstance(result, dict) else {"observation": str(result)}
            return StepRecord(
                step=step_number,
                goal=state.goal,
                action=action_name,
                observation=str(payload.get("observation", "")),
                success=bool(payload.get("success", True)),
                confidence=float(payload.get("confidence", 1.0)),
                progress_score=float(payload.get("progress_score", 0.0)),
                repeated_action=bool(payload.get("repeated_action", False)),
                contradiction=bool(payload.get("contradiction", False)),
                error=payload.get("error"),
                actual_failure=bool(payload.get("actual_failure", not payload.get("success", True))),
                failure_type=payload.get("failure_type"),
            )
        except Exception as exc:  # The trajectory should retain unexpected tool failures.
            return StepRecord(
                step=step_number,
                goal=state.goal,
                action=action_name,
                observation="Action raised an exception.",
                success=False,
                confidence=0.0,
                progress_score=0.0,
                error=f"{type(exc).__name__}: {exc}",
                actual_failure=True,
                failure_type="tool_failure",
            )
