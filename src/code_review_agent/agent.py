"""
CodeReviewAgent — dual-backend agentic loop (Ollama default / Anthropic optional).
Reads all settings from config.yaml via the config module.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from code_review_agent.config import get_config
from code_review_agent.prompts import SYSTEM_PROMPT
from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI, execute_tool

# ---------------------------------------------------------------------------
# Anthropic tool schemas (Claude format)
# ---------------------------------------------------------------------------

_ANTHROPIC_TOOLS: list[dict[str, Any]] = []
for _t in TOOL_DEFINITIONS_OPENAI:
    fn = _t["function"]
    _ANTHROPIC_TOOLS.append({
        "name": fn["name"],
        "description": fn["description"],
        "input_schema": fn["parameters"],
    })


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class _BaseAgent:
    def review(self, path: str, extra_context: str = "") -> Iterator[str]:
        prompt = f"Please perform a comprehensive code review of the project at: `{path}`"
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        yield from self._run(prompt)

    def ask(self, question: str) -> Iterator[str]:
        yield from self._run(question)

    def _run(self, user_prompt: str) -> Iterator[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ollama (OpenAI-compatible) backend
# ---------------------------------------------------------------------------

class OllamaAgent(_BaseAgent):
    """Uses Ollama via its OpenAI-compatible REST API."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        max_iterations: int,
        timeout: int = 120,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations

    def _run(self, user_prompt: str) -> Iterator[str]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS_OPENAI,
                tool_choice="auto",
                max_tokens=self.max_tokens,
                stream=True,
            )

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
                                tool_calls_map[idx]["function"]["arguments"] += tc.function.arguments

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
                yield f"\n\n*[Running tool: **{name}** …]*\n\n"
                result = execute_tool(name, inputs)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicAgent(_BaseAgent):
    """Uses Claude Opus 4.6 with adaptive thinking + tool use."""

    def __init__(self, api_key: str | None, model: str, max_tokens: int, max_iterations: int) -> None:
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations

    def _run(self, user_prompt: str) -> Iterator[str]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        for _ in range(self.max_iterations):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=_ANTHROPIC_TOOLS,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
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
                yield f"\n\n*[Running tool: **{tb.name}** …]*\n\n"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": execute_tool(tb.name, tb.input),
                })
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
        # Anthropic overrides
        api_key: str | None = None,
        model: str | None = None,
        # Ollama overrides
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        ollama_api_key: str | None = None,
        # Shared overrides
        max_tokens: int | None = None,
        max_iterations: int | None = None,
    ) -> _BaseAgent:
        cfg = get_config(config_path)
        provider = provider or cfg.provider

        if provider == "anthropic":
            return AnthropicAgent(
                api_key=api_key,
                model=model or cfg.anthropic.model,
                max_tokens=max_tokens or cfg.anthropic.max_tokens,
                max_iterations=max_iterations or cfg.anthropic.max_iterations,
            )

        # Default: ollama
        return OllamaAgent(
            model=ollama_model or cfg.ollama.model,
            base_url=ollama_url or cfg.ollama.base_url,
            api_key=ollama_api_key or cfg.ollama.api_key,
            max_tokens=max_tokens or cfg.ollama.max_tokens,
            max_iterations=max_iterations or cfg.ollama.max_iterations,
            timeout=cfg.ollama.timeout,
        )
