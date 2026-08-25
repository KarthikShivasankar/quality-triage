"""
LiteLLM client — the only LLM transport used by quality-triage.

Talks to local models (Ollama, vLLM, LM Studio) and frontier APIs
(OpenAI, Anthropic, Groq, Gemini, …) through one OpenAI-shaped interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from code_review_agent.config import (
    AppConfig,
    LLMConfig,
    as_litellm_ollama,
    has_litellm_prefix,
    ollama_native_name,
)

__all__ = [
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "ToolCall",
    "CompletionResult",
    "LLMClient",
    "resolve_model",
    "apply_provider_shim",
    "resolve_llm_model",
    "as_litellm_ollama",
    "ollama_native_name",
    "has_litellm_prefix",
]


class LLMError(Exception):
    """Base error for LiteLLM failures."""


class LLMAuthError(LLMError):
    """Missing or invalid API key."""


class LLMRateLimitError(LLMError):
    """Provider rate limit."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str | None = None


def resolve_model(
    name: str | None,
    aliases: dict[str, str],
    default: str,
) -> str:
    """Resolve an alias or pass-through LiteLLM model string."""
    raw = (name or default).strip()
    return aliases.get(raw, raw)


def apply_provider_shim(provider: str | None, model: str) -> str:
    """Prefix a bare model name with a legacy --provider value.

    Hugging Face GGUF tags like ``hf.co/org/repo:Q4_K_M`` contain slashes but
    are *not* LiteLLM provider prefixes, so they still get ``ollama/``.
    """
    if not provider:
        return model
    provider = provider.strip().lower()
    if has_litellm_prefix(model):
        return model
    if provider in ("ollama", "anthropic", "openai", "groq", "gemini", "azure"):
        return f"{provider}/{model}"
    return model


def resolve_llm_model(
    *,
    model: str | None,
    provider: str | None,
    cfg: AppConfig,
) -> str:
    """Resolve CLI overrides + aliases + optional legacy provider shim."""
    resolved = resolve_model(model, cfg.aliases, cfg.llm.model)
    if provider and not model:
        if provider == "anthropic":
            resolved = cfg.aliases.get("frontier", "anthropic/claude-sonnet-4-6")
        elif provider == "ollama":
            if resolved.startswith(("ollama/", "ollama_chat/")):
                pass
            elif has_litellm_prefix(resolved):
                resolved = as_litellm_ollama(cfg.aliases.get("local", resolved))
            else:
                resolved = as_litellm_ollama(resolved)
        else:
            resolved = apply_provider_shim(provider, resolved)
    elif provider and model:
        resolved = apply_provider_shim(provider, resolved)
    return resolved


def _delta_content(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return None
    c0 = choices[0]
    delta = getattr(c0, "delta", None)
    if delta is None and isinstance(c0, dict):
        delta = c0.get("delta")
    if delta is None:
        return None
    content = getattr(delta, "content", None)
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    return content or None


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    raw = getattr(message, "tool_calls", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("tool_calls")
    if not raw:
        return []
    out: list[ToolCall] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function") or {}
            tc_id = tc.get("id") or ""
            name = fn.get("name") or ""
            args_raw = fn.get("arguments") or "{}"
        else:
            tc_id = getattr(tc, "id", "") or ""
            name = getattr(fn, "name", "") or ""
            args_raw = getattr(fn, "arguments", None) or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCall(id=str(tc_id), name=str(name), arguments=args))
    return out


class LLMClient:
    """Thin, testable wrapper around litellm.completion."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        model: str | None = None,
        api_base: str | None = None,
        fallbacks: list[str] | None = None,
    ) -> None:
        self.model = model or cfg.model
        self.api_base = api_base if api_base is not None else cfg.api_base
        self.api_key = cfg.api_key
        self.timeout = cfg.timeout
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.num_retries = cfg.num_retries
        self.drop_params = cfg.drop_params
        self.fallbacks = list(fallbacks if fallbacks is not None else cfg.fallbacks)

    def _kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "num_retries": self.num_retries,
            "drop_params": self.drop_params,
            "stream": stream,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.fallbacks:
            kwargs["fallbacks"] = self.fallbacks
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _call(self, **kwargs: Any) -> Any:
        import litellm

        try:
            return litellm.completion(**kwargs)
        except litellm.AuthenticationError as exc:
            raise LLMAuthError(str(exc)) from exc
        except litellm.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except litellm.APIError as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        resp = self._call(**self._kwargs(messages, tools, stream=False))
        choices = getattr(resp, "choices", None)
        if choices is None and isinstance(resp, dict):
            choices = resp.get("choices")
        choice0 = choices[0]
        if isinstance(choice0, dict):
            message = choice0.get("message") or choice0
            finish = choice0.get("finish_reason")
        else:
            message = getattr(choice0, "message", None) or choice0
            finish = getattr(choice0, "finish_reason", None)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        model = getattr(resp, "model", None)
        if model is None and isinstance(resp, dict):
            model = resp.get("model")
        model = model or self.model
        return CompletionResult(
            content=content or "",
            tool_calls=_parse_tool_calls(message),
            model=str(model),
            finish_reason=str(finish) if finish else None,
        )

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        resp = self._call(**self._kwargs(messages, tools, stream=True))
        for chunk in resp:
            text = _delta_content(chunk)
            if text:
                yield text
