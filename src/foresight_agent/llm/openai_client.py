"""Optional OpenAI Responses API adapter."""

import os
import time

from foresight_agent.telemetry import LLMTelemetryLogger, LLMTrace
from .base import LLMResponse, Usage


class OpenAIResponsesClient:
    def __init__(
        self,
        model: str | None = None,
        telemetry: LLMTelemetryLogger | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[llm]'") from exc
        normalized_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not normalized_key:
            raise RuntimeError("Set OPENAI_API_KEY before creating the OpenAI client.")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        # Passing the normalized value prevents invisible shell whitespace from
        # becoming an invalid HTTP Authorization header.
        self._client, self.telemetry = OpenAI(api_key=normalized_key), telemetry

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.responses.create(model=self.model, instructions=system_prompt, input=user_prompt)
        elapsed = (time.perf_counter() - started) * 1000
        raw = getattr(response, "usage", None)
        usage = Usage(int(getattr(raw, "input_tokens", 0) or 0), int(getattr(raw, "output_tokens", 0) or 0))
        result = LLMResponse(response.output_text, self.model, usage)
        if self.telemetry:
            self.telemetry.log(LLMTrace(system_prompt, user_prompt, result.text, self.model, usage, elapsed))
        return result
