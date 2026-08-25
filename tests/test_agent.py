"""Mocked LiteLLMAgent loop tests."""

from __future__ import annotations

from code_review_agent.agent import CodeReviewAgent, LiteLLMAgent, make_agent
from code_review_agent.config import LLMConfig, load_config
from code_review_agent.llm import CompletionResult, LLMClient, ToolCall


class FakeClient(LLMClient):
    def __init__(self, turns: list[CompletionResult]):
        super().__init__(LLMConfig(model="fake/model"))
        self.turns = list(turns)
        self.calls = 0

    def complete_text(self, messages, tools=None):
        self.calls += 1
        if not self.turns:
            return CompletionResult(content="done")
        return self.turns.pop(0)


def test_agent_plain_answer():
    client = FakeClient([CompletionResult(content="looks fine")])
    agent = LiteLLMAgent(client, max_iterations=3)
    text = "".join(agent.ask("how is this file?"))
    assert "looks fine" in text
    assert client.calls == 1


def test_agent_runs_tool_then_finishes(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    turns = [
        CompletionResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="list_python_files",
                    arguments={"directory": str(tmp_path)},
                )
            ],
        ),
        CompletionResult(content="found 1 file"),
    ]
    client = FakeClient(turns)
    agent = LiteLLMAgent(client, max_iterations=5)
    text = "".join(agent.ask("list files"))
    assert "Running tool" in text
    assert "found 1 file" in text
    assert client.calls == 2


def test_make_agent_and_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    agent = make_agent(cfg, model="local")
    assert isinstance(agent, LiteLLMAgent)
    assert agent.client.model == cfg.aliases["local"]

    wrapped = CodeReviewAgent(provider="ollama", model="gemma3:latest")
    assert isinstance(wrapped, LiteLLMAgent)
    assert wrapped.client.model.startswith("ollama/")

    hf = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
    wrapped_hf = CodeReviewAgent(provider="ollama", model=hf)
    assert wrapped_hf.client.model == f"ollama/{hf}"
