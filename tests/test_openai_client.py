import sys
import types

from foresight_agent.llm.openai_client import OpenAIResponsesClient


def test_openai_client_strips_key_whitespace(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "  test-key\n")
    OpenAIResponsesClient(model="test-model")
    assert captured["api_key"] == "test-key"


def test_explicit_key_is_supported_and_normalized(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAIResponsesClient(model="test-model", api_key="  sk-test  ")
    assert captured["api_key"] == "sk-test"
