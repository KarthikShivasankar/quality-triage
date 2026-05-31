"""
Tests for code_review_agent.agent — fully offline.

The OpenAI client is replaced with a scripted fake so no network calls happen.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fake OpenAI streaming primitives
# ---------------------------------------------------------------------------

def _text_chunk(text):
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_chunk(call_id, name, arguments, index=0):
    tc = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    delta = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeCompletions:
    """Returns scripted streams; optionally raises a max_tokens error first."""

    def __init__(self, scripts, raise_token_error_once=False):
        self._scripts = list(scripts)
        self.calls = []
        self._raise_token_error_once = raise_token_error_once

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_token_error_once:
            self._raise_token_error_once = False
            raise Exception(
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        if not self._scripts:
            return iter([_text_chunk("")])
        return iter(self._scripts.pop(0))


class FakeClient:
    def __init__(self, completions, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=completions)
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))


def _install_fake_openai(monkeypatch, completions, capture=None):
    import openai

    def _factory(**kwargs):
        client = FakeClient(completions, **kwargs)
        if capture is not None:
            capture["client"] = client
        return client

    monkeypatch.setattr(openai, "OpenAI", _factory)


# ---------------------------------------------------------------------------
# Factory: correct class per provider
# ---------------------------------------------------------------------------

class TestFactory:
    def test_ollama_returns_openai_compatible(self, monkeypatch):
        _install_fake_openai(monkeypatch, FakeCompletions([]))
        from code_review_agent.agent import CodeReviewAgent, OpenAICompatibleAgent
        agent = CodeReviewAgent(provider="ollama")
        assert isinstance(agent, OpenAICompatibleAgent)

    def test_openai_returns_openai_compatible(self, monkeypatch):
        _install_fake_openai(monkeypatch, FakeCompletions([]))
        from code_review_agent.agent import CodeReviewAgent, OpenAICompatibleAgent
        agent = CodeReviewAgent(provider="openai", api_key="sk-test")
        assert isinstance(agent, OpenAICompatibleAgent)

    def test_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _install_fake_openai(monkeypatch, FakeCompletions([]))
        from code_review_agent.agent import CodeReviewAgent
        with pytest.raises(ValueError, match="(?i)api key"):
            CodeReviewAgent(provider="openai")

    def test_openai_base_url_and_key_resolution(self, monkeypatch):
        capture = {}
        _install_fake_openai(monkeypatch, FakeCompletions([]), capture=capture)
        from code_review_agent.agent import CodeReviewAgent
        CodeReviewAgent(
            provider="openai",
            api_key="sk-abc",
            base_url="https://example.test/v1",
        )
        assert capture["client"].init_kwargs["api_key"] == "sk-abc"
        assert capture["client"].init_kwargs["base_url"] == "https://example.test/v1"

    def test_openai_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        capture = {}
        _install_fake_openai(monkeypatch, FakeCompletions([]), capture=capture)
        from code_review_agent.agent import CodeReviewAgent
        CodeReviewAgent(provider="openai")
        assert capture["client"].init_kwargs["api_key"] == "sk-from-env"

    def test_anthropic_returns_anthropic_agent(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from code_review_agent.agent import CodeReviewAgent, AnthropicAgent
        agent = CodeReviewAgent(provider="anthropic")
        assert isinstance(agent, AnthropicAgent)

    def test_anthropic_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from code_review_agent.agent import CodeReviewAgent
        with pytest.raises(ValueError, match="API key"):
            CodeReviewAgent(provider="anthropic")


# ---------------------------------------------------------------------------
# Streaming + tool-call loop
# ---------------------------------------------------------------------------

class TestOpenAICompatibleAgent:
    def test_plain_text_response(self, monkeypatch):
        completions = FakeCompletions([[_text_chunk("Hello "), _text_chunk("world")]])
        _install_fake_openai(monkeypatch, completions)
        from code_review_agent.agent import CodeReviewAgent
        agent = CodeReviewAgent(provider="ollama")
        out = "".join(agent.ask("hi"))
        assert "Hello world" in out

    def test_tool_call_then_text(self, monkeypatch, tmp_path):
        # Run a real (offline) tool: list_python_files on a temp dir.
        (tmp_path / "a.py").write_text("x = 1")
        args = json.dumps({"directory": str(tmp_path)})
        completions = FakeCompletions([
            [_tool_chunk("call_1", "list_python_files", args)],
            [_text_chunk("Done analyzing.")],
        ])
        _install_fake_openai(monkeypatch, completions)
        from code_review_agent.agent import CodeReviewAgent
        agent = CodeReviewAgent(provider="ollama")
        out = "".join(agent.ask("review"))
        assert "Running tool" in out
        assert "list_python_files" in out
        assert "Done analyzing." in out
        # Two create() calls: one with the tool call, one after the tool result.
        assert len(completions.calls) == 2
        # The tool result was appended as a 'tool' role message.
        second_messages = completions.calls[1]["messages"]
        assert any(m.get("role") == "tool" for m in second_messages)

    def test_max_tokens_retry_path(self, monkeypatch):
        completions = FakeCompletions(
            [[_text_chunk("recovered")]],
            raise_token_error_once=True,
        )
        _install_fake_openai(monkeypatch, completions)
        from code_review_agent.agent import CodeReviewAgent
        agent = CodeReviewAgent(provider="openai", api_key="sk-x")
        out = "".join(agent.ask("hi"))
        assert "recovered" in out
        assert agent._token_param == "max_completion_tokens"
        # First (failed) call used max_tokens, retry used max_completion_tokens.
        assert "max_tokens" in completions.calls[0]
        assert "max_completion_tokens" in completions.calls[1]

    def test_extra_headers_passed(self, monkeypatch):
        capture = {}
        _install_fake_openai(monkeypatch, FakeCompletions([]), capture=capture)
        from code_review_agent.agent import OpenAICompatibleAgent
        OpenAICompatibleAgent(
            model="m", base_url="u", api_key="k",
            max_tokens=10, max_iterations=1,
            extra_headers={"X-Title": "t"},
        )
        assert capture["client"].init_kwargs["default_headers"] == {"X-Title": "t"}
