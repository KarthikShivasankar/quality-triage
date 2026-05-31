"""
CodeReviewAgent — multi-backend agentic loop.

Providers:
  * ``ollama``    — local Ollama via its OpenAI-compatible API (default, no key)
  * ``openai``    — any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter,
                    Together, Fireworks, Mistral, llama.cpp server, vLLM, LM Studio)
  * ``anthropic`` — Anthropic Claude

Both ``ollama`` and ``openai`` are served by a single ``OpenAICompatibleAgent``.
All settings come from config.yaml via the config module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from code_review_agent.config import get_config, resolve_api_key
from code_review_agent.prompts import SYSTEM_PROMPT
from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI, execute_tool

# ---------------------------------------------------------------------------
# Anthropic tool schemas (Claude format)
# ---------------------------------------------------------------------------

_ANTHROPIC_TOOLS: list[dict[str, Any]] = []
for _t in TOOL_DEFINITIONS_OPENAI:
    fn = _t["function"]
    _ANTHROPIC_TOOLS.append(
        {
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        }
    )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class _BaseAgent:
    def review(self, path: str, extra_context: str = "") -> Iterator[str]:
        prompt = f"Please perform a comprehensive code review of the project at: `{path}`"
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        yield from self._run(prompt)

    def review_text(self, path: str, extra_context: str = "") -> str:
        """Non-streaming convenience used by the web UI."""
        return "".join(self.review(path, extra_context=extra_context))

    def ask(self, question: str) -> Iterator[str]:
        yield from self._run(question)

    def ask_text(self, question: str) -> str:
        return "".join(self.ask(question))

    def _run(self, user_prompt: str) -> Iterator[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible backend (serves both `ollama` and `openai`)
# ---------------------------------------------------------------------------


class OpenAICompatibleAgent(_BaseAgent):
    """
    Agentic loop over any OpenAI-compatible ``/chat/completions`` endpoint.

    Used by both the ``ollama`` provider (local) and the ``openai`` provider
    (OpenAI, Groq, OpenRouter, Together, Fireworks, Mistral, llama.cpp server,
    vLLM, LM Studio, …) — they differ only by base_url, api_key, and headers.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        max_tokens: int,
        max_iterations: int,
        timeout: int = 120,
        extra_headers: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
        allowed_tool_names: set[str] | None = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            default_headers=extra_headers or None,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        # Optional selection: restrict which tool schemas the model sees and
        # which tools it is allowed to actually call.
        self.tools = tools if tools is not None else TOOL_DEFINITIONS_OPENAI
        self.allowed_tool_names = set(allowed_tool_names) if allowed_tool_names else None
        # Some endpoints (newer OpenAI models) reject `max_tokens` and require
        # `max_completion_tokens`. We flip this on the fly if asked to.
        self._token_param = "max_tokens"

    def _create_stream(self, messages: list[dict[str, Any]]):
        """Start a streaming completion, retrying once on a max_tokens param mismatch."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            "stream": True,
            self._token_param: self.max_tokens,
        }
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Swap max_tokens <-> max_completion_tokens and retry once.
            if ("max_tokens" in msg or "max_completion_tokens" in msg) and (
                "unsupported" in msg
                or "not supported" in msg
                or "use" in msg
                or "invalid" in msg
                or "max_completion_tokens" in msg
            ):
                new_param = (
                    "max_completion_tokens" if self._token_param == "max_tokens" else "max_tokens"
                )
                kwargs.pop(self._token_param, None)
                kwargs[new_param] = self.max_tokens
                self._token_param = new_param
                return self.client.chat.completions.create(**kwargs)
            raise

    def _run(self, user_prompt: str) -> Iterator[str]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.max_iterations):
            response = self._create_stream(messages)

            full_content = ""
            tool_calls_map: dict[int, dict[str, Any]] = {}

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                if delta.content:
                    full_content += delta.content
                    yield delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["function"]["arguments"] += (
                                    tc.function.arguments
                                )

            tool_calls = list(tool_calls_map.values())
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    inputs = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    inputs = {}
                if self.allowed_tool_names is not None and name not in self.allowed_tool_names:
                    result = json.dumps(
                        {
                            "error": f"Tool '{name}' is not enabled for this review selection.",
                            "allowed_tools": sorted(self.allowed_tool_names),
                        }
                    )
                    yield f"\n\n*[Tool **{name}** skipped — not in selection]*\n\n"
                else:
                    yield f"\n\n*[Running tool: **{name}** …]*\n\n"
                    result = execute_tool(name, inputs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )


# Backwards-compatible alias.
OllamaAgent = OpenAICompatibleAgent


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


class AnthropicAgent(_BaseAgent):
    """Uses Claude Opus 4.6 with adaptive thinking + tool use."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_tokens: int,
        max_iterations: int,
        tools: list[dict[str, Any]] | None = None,
        allowed_tool_names: set[str] | None = None,
    ) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic provider requires an API key. Set ANTHROPIC_API_KEY "
                "or anthropic.api_key in config.yaml."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        if tools is not None:
            self.tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        else:
            self.tools = _ANTHROPIC_TOOLS
        self.allowed_tool_names = set(allowed_tool_names) if allowed_tool_names else None

    def _run(self, user_prompt: str) -> Iterator[str]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        for _ in range(self.max_iterations):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=self.tools,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                response = stream.get_final_message()

            content = response.content
            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_use_blocks = [b for b in content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            messages.append({"role": "assistant", "content": content})
            tool_results = []
            for tb in tool_use_blocks:
                if self.allowed_tool_names is not None and tb.name not in self.allowed_tool_names:
                    yield f"\n\n*[Tool **{tb.name}** skipped — not in selection]*\n\n"
                    result = json.dumps(
                        {
                            "error": f"Tool '{tb.name}' is not enabled for this review selection.",
                        }
                    )
                else:
                    yield f"\n\n*[Running tool: **{tb.name}** …]*\n\n"
                    result = execute_tool(tb.name, tb.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class CodeReviewAgent:
    """
    Factory returning the correct backend based on config.yaml `provider`.

    Override via constructor kwargs or CLI flags.
    Defaults come entirely from config.yaml.
    """

    def __new__(
        cls,
        provider: str | None = None,
        config_path: str | None = None,
        # Anthropic / openai overrides
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        # Ollama overrides
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        ollama_api_key: str | None = None,
        # Shared overrides
        max_tokens: int | None = None,
        max_iterations: int | None = None,
        # Analysis selection (restrict the tool surface offered to the model)
        tools: list[dict[str, Any]] | None = None,
        allowed_tool_names: set[str] | None = None,
    ) -> _BaseAgent:
        cfg = get_config(config_path)
        provider = provider or cfg.provider

        if provider == "anthropic":
            return AnthropicAgent(
                api_key=resolve_api_key(
                    api_key or cfg.anthropic.api_key,
                    cfg.anthropic.api_key_env,
                    fallback_env="ANTHROPIC_API_KEY",
                ),
                model=model or cfg.anthropic.model,
                max_tokens=max_tokens or cfg.anthropic.max_tokens,
                max_iterations=max_iterations or cfg.anthropic.max_iterations,
                tools=tools,
                allowed_tool_names=allowed_tool_names,
            )

        if provider == "openai":
            key = resolve_api_key(
                api_key or cfg.openai.api_key,
                cfg.openai.api_key_env,
                fallback_env="OPENAI_API_KEY",
            )
            resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or cfg.openai.base_url
            if not key:
                raise ValueError(
                    f"openai provider requires an API key. Set the "
                    f"'{cfg.openai.api_key_env}' environment variable (or OPENAI_API_KEY), "
                    f"or pass --api-key."
                )
            return OpenAICompatibleAgent(
                model=model or cfg.openai.model,
                base_url=resolved_base,
                api_key=key,
                max_tokens=max_tokens or cfg.openai.max_tokens,
                max_iterations=max_iterations or cfg.openai.max_iterations,
                timeout=cfg.openai.timeout,
                extra_headers=cfg.openai.extra_headers or None,
                tools=tools,
                allowed_tool_names=allowed_tool_names,
            )

        # Default: ollama (local, OpenAI-compatible)
        return OpenAICompatibleAgent(
            model=ollama_model or model or cfg.ollama.model,
            base_url=ollama_url or base_url or cfg.ollama.base_url,
            api_key=ollama_api_key or cfg.ollama.api_key,
            max_tokens=max_tokens or cfg.ollama.max_tokens,
            max_iterations=max_iterations or cfg.ollama.max_iterations,
            timeout=cfg.ollama.timeout,
            tools=tools,
            allowed_tool_names=allowed_tool_names,
        )
