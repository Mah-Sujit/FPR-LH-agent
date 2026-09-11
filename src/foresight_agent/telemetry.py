import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from foresight_agent.llm.base import Usage


@dataclass
class LLMTrace:
    system_prompt: str
    user_prompt: str
    response: str
    model: str
    usage: Usage
    latency_ms: float
    estimated_cost_usd: float | None = None


class LLMTelemetryLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, trace: LLMTrace) -> None:
        payload = asdict(trace)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
