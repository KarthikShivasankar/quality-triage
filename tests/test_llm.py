"""Unit tests for the LiteLLM wrapper (mocked completion)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code_review_agent.config import AppConfig, LLMConfig, load_config
from code_review_agent.llm import (
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMRateLimitError,
    apply_provider_shim,
    as_litellm_ollama,
    ollama_native_name,
    resolve_llm_model,
    resolve_model,
)


class _Fn(SimpleNamespace):
    pass


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(content="", tool_calls=None, model="ollama/x"):
    choice = SimpleNamespace(
        message=_message(content, tool_calls), finish_reason="stop"
    )
    return SimpleNamespace(choices=[choice], model=model)


def test_resolve_model_alias():
    assert resolve_model("local", {"local": "ollama/x"}, "y") == "ollama/x"
    assert (
        resolve_model("openai/gpt-4.1", {"local": "ollama/x"}, "x") == "openai/gpt-4.1"
    )
    assert resolve_model(None, {}, "ollama/default") == "ollama/default"


def test_apply_provider_shim():
    hf = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
    assert apply_provider_shim("ollama", "gemma3:latest") == "ollama/gemma3:latest"
    assert (
        apply_provider_shim("anthropic", "claude-sonnet-4-6")
        == "anthropic/claude-sonnet-4-6"
    )
    assert apply_provider_shim("ollama", "openai/gpt-4.1") == "openai/gpt-4.1"
    assert apply_provider_shim("ollama", hf) == f"ollama/{hf}"
    assert apply_provider_shim("ollama", f"ollama/{hf}") == f"ollama/{hf}"
    assert apply_provider_shim(None, "x") == "x"


def test_ollama_name_helpers():
    hf = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
    assert as_litellm_ollama(hf) == f"ollama/{hf}"
    assert as_litellm_ollama(f"ollama/{hf}") == f"ollama/{hf}"
    assert ollama_native_name(f"ollama/{hf}") == hf
    assert ollama_native_name(hf) == hf


def test_resolve_llm_model_hf_gguf_with_provider():
    cfg = AppConfig()
    hf = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
    assert resolve_llm_model(model=hf, provider="ollama", cfg=cfg) == f"ollama/{hf}"
    assert (
        resolve_llm_model(model="test", provider=None, cfg=cfg) == cfg.aliases["test"]
    )


def test_resolve_llm_model_provider_only_ollama_keeps_hf_default():
    cfg = AppConfig()
    model = resolve_llm_model(model=None, provider="ollama", cfg=cfg)
    assert model.startswith("ollama/hf.co/")


def test_resolve_llm_model_provider_only_anthropic():
    cfg = AppConfig()
    model = resolve_llm_model(model=None, provider="anthropic", cfg=cfg)
    assert model.startswith("anthropic/")


def test_resolve_llm_model_cli_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert (
        resolve_llm_model(model="cheap", provider=None, cfg=cfg) == cfg.aliases["cheap"]
    )


def test_complete_text_parses_content(monkeypatch):
    client = LLMClient(LLMConfig(model="ollama/x"))

    def fake_completion(**kwargs):
        assert kwargs["model"] == "ollama/x"
        assert kwargs["drop_params"] is True
        assert kwargs["stream"] is False
        return _response("hello")

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = client.complete_text([{"role": "user", "content": "hi"}])
    assert result.content == "hello"
    assert result.tool_calls == []


def test_complete_text_parses_tool_calls(monkeypatch):
    client = LLMClient(LLMConfig(model="openai/gpt-4.1"))
    tc = SimpleNamespace(
        id="call_1",
        function=_Fn(name="list_python_files", arguments='{"directory": "/tmp"}'),
    )

    def fake_completion(**kwargs):
        assert "tools" in kwargs
        return _response("", tool_calls=[tc])

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = client.complete_text(
        [{"role": "user", "content": "go"}], tools=[{"type": "function"}]
    )
    assert result.tool_calls[0].name == "list_python_files"
    assert result.tool_calls[0].arguments == {"directory": "/tmp"}


def test_complete_text_bad_tool_json(monkeypatch):
    client = LLMClient(LLMConfig())
    tc = SimpleNamespace(id="x", function=_Fn(name="read_file", arguments="not-json"))

    monkeypatch.setattr(
        "litellm.completion", lambda **k: _response("", tool_calls=[tc])
    )
    result = client.complete_text([{"role": "user", "content": "x"}])
    assert result.tool_calls[0].arguments == {}


def test_stream_text(monkeypatch):
    client = LLMClient(LLMConfig())

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]
            ),
        ]
        return iter(chunks)

    monkeypatch.setattr("litellm.completion", fake_completion)
    assert "".join(client.stream_text([{"role": "user", "content": "x"}])) == "Hello"


def test_auth_error(monkeypatch):
    import litellm

    class DummyAuth(Exception):
        pass

    monkeypatch.setattr(litellm, "AuthenticationError", DummyAuth)

    def boom(**kwargs):
        raise DummyAuth("no key")

    monkeypatch.setattr(litellm, "completion", boom)
    client = LLMClient(LLMConfig())
    with pytest.raises(LLMAuthError):
        client.complete_text([{"role": "user", "content": "x"}])


def test_generic_error(monkeypatch):
    client = LLMClient(LLMConfig())
    monkeypatch.setattr(
        "litellm.completion", lambda **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    with pytest.raises(LLMError, match="down"):
        client.complete_text([{"role": "user", "content": "x"}])


def test_fallbacks_and_api_base_passed(monkeypatch):
    captured = {}
    cfg = LLMConfig(
        model="ollama/x", api_base="http://localhost:11434", fallbacks=["groq/y"]
    )
    client = LLMClient(cfg, fallbacks=["openai/gpt-4.1"])

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _response("ok")

    monkeypatch.setattr("litellm.completion", fake_completion)
    client.complete_text([{"role": "user", "content": "x"}])
    assert captured["fallbacks"] == ["openai/gpt-4.1"]
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["num_retries"] == 2


def test_rate_limit_error(monkeypatch):
    import litellm

    class DummyRate(Exception):
        pass

    monkeypatch.setattr(litellm, "RateLimitError", DummyRate)
    monkeypatch.setattr(
        litellm, "completion", lambda **k: (_ for _ in ()).throw(DummyRate("slow down"))
    )
    client = LLMClient(LLMConfig())
    with pytest.raises(LLMRateLimitError):
        client.complete_text([{"role": "user", "content": "x"}])


def test_complete_text_dict_message(monkeypatch):
    client = LLMClient(LLMConfig(model="ollama/x"))

    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {"content": "pong", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "model": "ollama/x",
        }

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = client.complete_text([{"role": "user", "content": "ping"}])
    assert result.content == "pong"


def test_complete_text_dict_tool_calls(monkeypatch):
    client = LLMClient(LLMConfig())
    monkeypatch.setattr(
        "litellm.completion",
        lambda **k: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"file_path": "a.py"}',
                                },
                            }
                        ],
                    },
                    finish_reason="tool_calls",
                )
            ],
            model="x",
        ),
    )
    result = client.complete_text([{"role": "user", "content": "x"}])
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments["file_path"] == "a.py"
