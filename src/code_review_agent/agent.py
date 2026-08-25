"""
LiteLLMAgent — single agentic loop used by `ask` and `review --agentic`.

All provider differences are handled by LiteLLM.
"""

from __future__ import annotations

import json
from typing import Iterator

from code_review_agent.config import AppConfig, get_config
from code_review_agent.llm import LLMClient, resolve_llm_model
from code_review_agent.prompts import SYSTEM_PROMPT
from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI, execute_tool


class LiteLLMAgent:
    def __init__(
        self,
        client: LLMClient,
        max_iterations: int = 20,
    ) -> None:
        self.client = client
        self.max_iterations = max_iterations

    def review(self, path: str, extra_context: str = "") -> Iterator[str]:
        prompt = (
            f"Please perform a comprehensive code review of the project at: `{path}`"
        )
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        yield from self._run(prompt)

    def ask(self, question: str) -> Iterator[str]:
        yield from self._run(question)

    def _run(self, user_prompt: str) -> Iterator[str]:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.max_iterations):
            result = self.client.complete_text(messages, tools=TOOL_DEFINITIONS_OPENAI)
            if result.content:
                yield result.content
            if not result.tool_calls:
                break

            assistant_msg: dict = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in result.tool_calls:
                yield f"\n\n*[Running tool: **{tc.name}** …]*\n\n"
                tool_result = execute_tool(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )


def make_agent(
    cfg: AppConfig | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    api_base: str | None = None,
    fallbacks: list[str] | None = None,
    config_path: str | None = None,
) -> LiteLLMAgent:
    """Build a LiteLLMAgent from config + CLI overrides."""
    cfg = cfg or get_config(config_path)
    resolved = resolve_llm_model(model=model, provider=provider, cfg=cfg)
    client = LLMClient(
        cfg.llm,
        model=resolved,
        api_base=api_base,
        fallbacks=fallbacks,
    )
    return LiteLLMAgent(client, max_iterations=cfg.llm.max_iterations)


class CodeReviewAgent:
    """
    Back-compat factory. Returns a LiteLLMAgent.

    Historical dual backends (OllamaAgent / AnthropicAgent) are gone;
    LiteLLM routes every provider.
    """

    def __new__(cls, *args, **kwargs) -> LiteLLMAgent:  # type: ignore[misc]
        cfg = kwargs.pop("config_path", None)
        if cfg:
            from code_review_agent.config import get_config

            get_config(cfg)
        # Map legacy constructor kwargs
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None) or kwargs.pop("ollama_model", None)
        api_base = kwargs.pop("api_base", None) or kwargs.pop("ollama_url", None)
        fallbacks = kwargs.pop("fallbacks", None)
        return make_agent(
            model=model,
            provider=provider,
            api_base=api_base,
            fallbacks=fallbacks,
            config_path=cfg if isinstance(cfg, str) else None,
        )
