from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = Usage()


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...
