import json

from foresight_agent.llm.base import Usage
from foresight_agent.telemetry import LLMTelemetryLogger, LLMTrace


def test_telemetry_writes_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    logger = LLMTelemetryLogger(path)
    logger.log(LLMTrace("system", "user", "answer", "mock", Usage(3, 2), 12.5))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["usage"]["input_tokens"] == 3
    assert saved["latency_ms"] == 12.5
    assert "timestamp" in saved
