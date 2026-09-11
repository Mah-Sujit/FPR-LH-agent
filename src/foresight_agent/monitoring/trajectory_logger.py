"""In-memory and optional JSON Lines trajectory logging."""

from __future__ import annotations

import json
from pathlib import Path

from foresight_agent.agent.state import AgentState, StepRecord


class TrajectoryLogger:
    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path else None

    def log(self, state: AgentState, record: StepRecord) -> None:
        state.trajectory.append(record)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

