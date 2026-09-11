from collections.abc import Iterable

from .base import LLMResponse, Usage


class MockLLMClient:
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        text = next(self._responses)
        return LLMResponse(text, "mock-model", Usage(len(user_prompt.split()), len(text.split())))
