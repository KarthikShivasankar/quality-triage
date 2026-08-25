"""Tests for the deterministic analysis pipeline."""

from __future__ import annotations

from code_review_agent.config import LLMConfig, load_config
from code_review_agent.llm import CompletionResult, LLMClient
from code_review_agent.pipeline import (
    clean_synthesis,
    execute_hybrid_review,
    extract_debt_comments,
    run_pipeline,
    stamp_duration,
    stream_synthesis,
    synthesize_report,
)
from code_review_agent.reporter import build_report


def _sample(tmp_path):
    (tmp_path / "main.py").write_text(
        "import os\n\n"
        "# TODO: refactor this later — quick hack\n"
        "def long_function():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    (tmp_path / "utils.py").write_text(
        "# FIXME: this function has a security risk\n"
        "def helper(a, b):\n"
        "    return a * b\n"
    )
    return tmp_path


def test_extract_debt_comments(tmp_path):
    _sample(tmp_path)
    texts = extract_debt_comments(str(tmp_path), ignore=set())
    assert any("TODO" in t for t in texts)
    assert any("FIXME" in t for t in texts)


def test_extract_debt_comments_limit(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("\n".join(f"# TODO: item {i}" for i in range(20)))
    texts = extract_debt_comments(str(tmp_path), ignore=set(), limit=3)
    assert len(texts) == 3


def test_run_pipeline_no_llm(tmp_path, monkeypatch):
    _sample(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()

    def fake_td(texts, **kwargs):
        return {
            "tool": "td_classify",
            "predictions": [
                {"text": texts[0], "predicted_class": 2, "predicted_probability": 0.9}
            ],
        }

    monkeypatch.setattr("code_review_agent.pipeline.classify_technical_debt", fake_td)
    result = run_pipeline(str(tmp_path), cfg=cfg, parallel=False)
    assert result.report.files_analyzed >= 2
    assert "list_python_files" in result.report.tools_run
    assert "analyze_code_intelligence" in result.report.tools_run
    assert result.intel_raw.get("tool") == "code_intel"
    assert result.files_raw.get("total_files") >= 2


def test_normalize_and_run_pipeline_subset(tmp_path, monkeypatch):
    from code_review_agent.pipeline import normalize_pipeline_tools

    assert normalize_pipeline_tools(None)[-1] == "classify-td"
    assert normalize_pipeline_tools(["ml-smells", "bogus"]) == ["ml-smells"]
    assert normalize_pipeline_tools([]) == []

    _sample(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    called = {"td": 0, "ml": 0, "py": 0}

    monkeypatch.setattr(
        "code_review_agent.pipeline.classify_technical_debt",
        lambda *a, **k: (
            called.__setitem__("td", called["td"] + 1) or {"predictions": []}
        ),
    )
    monkeypatch.setattr(
        "code_review_agent.pipeline.detect_ml_smells",
        lambda *a, **k: called.__setitem__("ml", called["ml"] + 1) or {"tool": "ml"},
    )
    monkeypatch.setattr(
        "code_review_agent.pipeline.detect_python_smells",
        lambda *a, **k: called.__setitem__("py", called["py"] + 1) or {"tool": "py"},
    )
    result = run_pipeline(
        str(tmp_path),
        cfg=cfg,
        parallel=False,
        tools=["list-files", "ml-smells"],
    )
    assert called["ml"] == 1
    assert called["td"] == 0
    assert called["py"] == 0
    assert "detect_ml_smells" in result.report.tools_run
    assert "detect_python_smells" not in result.report.tools_run
    assert "classify_technical_debt" not in result.report.tools_run


def test_run_pipeline_requires_a_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    cfg = load_config()
    try:
        run_pipeline(str(tmp_path), cfg=cfg, parallel=False, tools=[])
    except ValueError as exc:
        assert "at least one" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_synthesize_report_uses_health_payload(monkeypatch):
    captured = {}

    class Fake(LLMClient):
        def complete_text(self, messages, tools=None):
            captured["messages"] = messages
            return CompletionResult(content="## Executive summary\nHealth 90")

    data = build_report(target="/proj", provider="ollama", model="x")
    text = synthesize_report(data, Fake(LLMConfig()))
    assert "Health 90" in text
    user = captured["messages"][1]["content"]
    assert "health_score" in user


def test_clean_synthesis_strips_think_preamble():
    raw = "I will plan the report.\n</think>\n## Executive summary\nHealth 80\n"
    assert clean_synthesis(raw) == "## Executive summary\nHealth 80"
    nested = (
        "## Executive summary\nOne short paragraph.\n\n"
        "## Executive summary\nThe real narrative"
    )
    assert clean_synthesis(nested) == "## Executive summary\nThe real narrative"
    assert clean_synthesis("") == ""


def test_synthesize_report_cleans_model_preamble():
    class Fake(LLMClient):
        def complete_text(self, messages, tools=None):
            return CompletionResult(
                content="thinking…</think>\n## Executive summary\nHealth 90"
            )

    data = build_report(target="/proj", provider="ollama", model="x")
    text = synthesize_report(data, Fake(LLMConfig()))
    assert text.startswith("## Executive summary")
    assert "thinking" not in text


def test_stream_synthesis(monkeypatch):
    class Fake(LLMClient):
        def stream_text(self, messages, tools=None):
            yield "Hel"
            yield "lo"

    data = build_report(target="/proj", provider="ollama", model="x")
    assert "".join(stream_synthesis(data, Fake(LLMConfig()))) == "Hello"


def test_stamp_duration():
    from datetime import datetime, timedelta, timezone

    data = build_report(target="/proj", provider="ollama", model="x")
    started = datetime.now(timezone.utc) - timedelta(seconds=2)
    stamp_duration(data, started)
    assert data.duration_s is not None
    assert data.duration_s >= 1.5


def test_execute_hybrid_review_no_llm(tmp_path, monkeypatch):
    _sample(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    monkeypatch.setattr(
        "code_review_agent.pipeline.classify_technical_debt",
        lambda texts, **k: {"predictions": []},
    )
    result = execute_hybrid_review(str(tmp_path), cfg, no_llm=True, parallel=False)
    assert result.report.narrative is None
    assert result.report.duration_s is not None
    assert result.synthesis_error is None


def test_execute_hybrid_review_synthesis_error(tmp_path, monkeypatch):
    _sample(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    monkeypatch.setattr(
        "code_review_agent.pipeline.classify_technical_debt",
        lambda texts, **k: {"predictions": []},
    )

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("code_review_agent.pipeline.synthesize_report", boom)
    result = execute_hybrid_review(str(tmp_path), cfg, no_llm=False, parallel=False)
    assert result.synthesis_error
    assert "ollama down" in result.synthesis_error


def test_run_pipeline_includes_github_issue_texts(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    captured: dict = {}

    def fake_td(texts, **kwargs):
        captured["texts"] = list(texts)
        return {
            "predictions": [
                {
                    "text": texts[0],
                    "predicted_class": 1,
                    "predicted_probability": 0.91,
                }
            ]
        }

    monkeypatch.setattr("code_review_agent.pipeline.classify_technical_debt", fake_td)
    result = run_pipeline(
        str(tmp_path),
        cfg=cfg,
        parallel=False,
        issue_texts=["#12 Fix the hacks in the CLI parser"],
    )
    assert captured["texts"] == ["#12 Fix the hacks in the CLI parser"]
    assert "classify_technical_debt" in result.report.tools_run
    assert result.report.td_predictions


def test_run_pipeline_merges_issues_with_todos(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("# TODO: local debt\nx = 1\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    captured: dict = {}

    def fake_td(texts, **kwargs):
        captured["texts"] = list(texts)
        return {"predictions": []}

    monkeypatch.setattr("code_review_agent.pipeline.classify_technical_debt", fake_td)
    run_pipeline(
        str(tmp_path),
        cfg=cfg,
        parallel=False,
        issue_texts=["#9 docs quality"],
    )
    assert any("TODO" in t for t in captured["texts"])
    assert "#9 docs quality" in captured["texts"]


def test_execute_hybrid_review_forwards_issue_texts(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    captured: dict = {}

    def fake_td(texts, **kwargs):
        captured["texts"] = list(texts)
        return {"predictions": []}

    monkeypatch.setattr("code_review_agent.pipeline.classify_technical_debt", fake_td)
    execute_hybrid_review(
        str(tmp_path),
        cfg,
        no_llm=True,
        parallel=False,
        issue_texts=["#1 opened recently"],
    )
    assert captured["texts"] == ["#1 opened recently"]
