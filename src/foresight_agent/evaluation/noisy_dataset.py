"""Seeded failure injection with intentionally overlapping risk signals."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep


@dataclass(frozen=True)
class FailureProfile:
    failure_type: str
    probability: float
    signal_recall: float
    false_alarm_rate: float
    recovery_probability: float
    latent: bool = False


@dataclass(frozen=True)
class NoisyCase:
    name: str
    failure_type: str
    build_planner: Callable[[int], Planner]

    @staticmethod
    def success_check(state: AgentState) -> bool:
        return state.memory.get("answer") == "correct"


PROFILES = (
    FailureProfile("tool_execution_error", 0.45, 0.90, 0.06, 0.85),
    FailureProfile("planning_error", 0.38, 0.72, 0.10, 0.70),
    FailureProfile("repetition_loop", 0.42, 0.75, 0.12, 0.78),
    FailureProfile("missing_evidence", 0.40, 0.62, 0.14, 0.72, True),
    FailureProfile("contradiction", 0.34, 0.68, 0.09, 0.74, True),
    FailureProfile("hallucination", 0.36, 0.48, 0.08, 0.65, True),
)


def _planner(profile: FailureProfile, seed: int) -> Planner:
    rng = random.Random(seed)

    def execute(state: AgentState):
        state.memory.pop("answer", None)
        recovering = bool(state.retries.get(state.current_step, 0) or
                          state.current_step in state.verified_steps or state.replans)
        probability = profile.probability * (1.0 - profile.recovery_probability if recovering else 1.0)
        failed = rng.random() < probability
        if failed:
            signaled = rng.random() < profile.signal_recall
            confidence = rng.uniform(0.12, 0.58) if signaled else rng.uniform(0.62, 0.96)
            contradiction = signaled and profile.failure_type in {"contradiction", "hallucination"}
            repeated = signaled and profile.failure_type in {"repetition_loop", "planning_error"}
            return {
                "observation": f"Noisy {profile.failure_type}",
                "success": profile.latent or not signaled,
                "confidence": confidence,
                # Progress is intentionally ambiguous: a failing step can appear
                # nearly complete, as often happens with latent agent errors.
                "progress_score": rng.uniform(0.15, 1.0),
                "repeated_action": repeated,
                "contradiction": contradiction,
                "actual_failure": True,
                "failure_type": profile.failure_type,
            }

        state.memory["answer"] = "correct"
        false_alarm = rng.random() < profile.false_alarm_rate
        return {
            "observation": "Correct result with noisy monitoring",
            "success": True,
            "confidence": rng.uniform(0.30, 0.60) if false_alarm else rng.uniform(0.78, 0.99),
            # Successful intermediate work need not report full completion.
            "progress_score": rng.uniform(0.65, 1.0),
            "repeated_action": false_alarm and rng.random() < 0.35,
            "contradiction": false_alarm and rng.random() < 0.12,
            "actual_failure": False,
        }

    def replan(state: AgentState):
        del state
        return [TaskStep("replanned_action", execute)]

    return Planner([TaskStep("noisy_action", execute)], replan_hook=replan)


def noisy_benchmarks(variants: int = 5) -> list[NoisyCase]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    cases = []
    for profile_index, profile in enumerate(PROFILES):
        for variant in range(1, variants + 1):
            offset = profile_index * 10_000 + variant * 101
            cases.append(NoisyCase(
                name=f"{profile.failure_type}_{variant:02d}",
                failure_type=profile.failure_type,
                build_planner=lambda seed, item=profile, extra=offset: _planner(item, seed + extra),
            ))
    return cases
