"""Gradio companion helpers (no live LLM)."""

from __future__ import annotations

import json
import socket
import urllib.request

import pytest

from code_review_agent.app import (
    _parse_pytest_terminal,
    cli_preview_markdown,
    equivalent_cli_review,
    equivalent_cli_tool,
    run_ask,
    run_review,
    run_tool,
    tool_preview_markdown,
)


def test_parse_pytest_terminal():
    log = (
        "tests/test_x.py::TestA::test_one PASSED\n"
        "tests/test_x.py::TestA::test_two FAILED\n"
        "tests/test_x.py::test_skip SKIPPED\n"
    )
    report = _parse_pytest_terminal(log)
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["skipped"] == 1
    assert report["summary"]["total"] == 3


def test_equivalent_cli_review():
    assert equivalent_cli_review("./src") == "code-review review ./src --no-llm"
    assert (
        equivalent_cli_review("./src", no_llm=False, model="frontier")
        == "code-review review ./src --model frontier"
    )
    assert (
        equivalent_cli_review("", no_llm=True) == "code-review review <path> --no-llm"
    )
    assert "local" not in equivalent_cli_review("./src", model="local")
    preview = cli_preview_markdown("./src", "local", True)
    assert "`code-review review ./src --no-llm`" in preview


def test_equivalent_cli_tool():
    assert equivalent_cli_tool("code-intel", "./src") == (
        "code-review run-tool code-intel ./src"
    )
    assert "--text" in equivalent_cli_tool("classify-td", "")
    assert "--from-file" in equivalent_cli_tool("td-from-comments", "notes.txt")
    assert "code-intel" in tool_preview_markdown("code-intel", "./src")


def test_run_tool_classify_td_requires_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = run_tool("classify-td", "", None, "", None)
    payload = json.loads(raw)
    assert "error" in payload
    assert "cli" in payload


def test_run_tool_list_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    raw = run_tool("list-files", str(tmp_path), None, "", None)
    payload = json.loads(raw)
    assert payload.get("total_files", 0) >= 1


def test_run_tool_empty_path_includes_cli():
    raw = run_tool("code-intel", "", None, "", None)
    payload = json.loads(raw)
    assert "error" in payload
    assert "code-review run-tool" in payload.get("cli", "")


def test_run_ask_empty_question():
    text = run_ask("", "local", None)
    assert "code-review ask" in text


def test_run_review_empty_path():
    chunks = list(run_review("", None, "local", True, "", None))
    assert chunks
    status, report_md, *_ = chunks[-1]
    assert status.startswith("Failed:")
    assert "code-review review" in report_md
    assert "<path>" in report_md or "`code-review review" in report_md


def test_run_review_pipeline_only(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    chunks = list(run_review(str(tmp_path), None, "local", True, "", None))
    assert len(chunks) >= 1
    status, report_md, rows, json_text, saved, *rest = chunks[-1]
    assert not status.startswith("Failed:"), status
    assert "# Code Review Report" in report_md
    assert "Health score:" in report_md
    payload = json.loads(json_text)
    assert "health_score" in payload
    assert isinstance(rows, list)
    assert saved
    assert ".md" in saved
    choices = rest[0] if rest else []
    assert isinstance(choices, list)
    assert any(path.endswith(".md") for _, path in choices)


def test_run_review_github_forwards_issue_texts(tmp_path, monkeypatch):
    from code_review_agent.github_utils import ClonedRepo, GithubIssue
    from code_review_agent.pipeline import PipelineResult
    from code_review_agent.reporter import build_report

    captured: dict = {}
    cloned = ClonedRepo(
        url="https://github.com/acme/demo",
        local_path=str(tmp_path),
        repo_name="acme/demo",
        branch="main",
        commit_sha="abc",
        is_temp=False,
        issues=[GithubIssue(number=4, title="Remove deprecated", body="now")],
    )

    def fake_resolve(target, **kwargs):
        captured["issue_limit"] = kwargs.get("issue_limit")
        return str(tmp_path), cloned

    def fake_exec(path, cfg, **kwargs):
        captured["issue_texts"] = kwargs.get("issue_texts")
        report = build_report(target=path, provider="ollama", model="x")
        report.narrative = None
        return PipelineResult(
            report=report,
            ml_raw={},
            py_raw={},
            td_raw={},
            intel_raw={},
            files_raw={},
        )

    monkeypatch.setattr(
        "code_review_agent.github_utils.resolve_review_target", fake_resolve
    )
    monkeypatch.setattr("code_review_agent.pipeline.execute_hybrid_review", fake_exec)
    monkeypatch.setattr(
        "code_review_agent.reporter.save_report", lambda *a, **k: ["/tmp/x.md"]
    )
    chunks = list(
        run_review("https://github.com/acme/demo", None, "local", True, "", None)
    )
    status = chunks[-1][0]
    assert not status.startswith("Failed:"), status
    assert captured["issue_limit"] == 10
    assert captured["issue_texts"][0].startswith("#4 Remove deprecated")


def test_build_ui_imports():
    pytest.importorskip("gradio")
    from code_review_agent.app import build_ui

    demo = build_ui()
    assert demo is not None
    assert demo.title == "Quality Triage"


def test_load_saved_report(tmp_path):
    from code_review_agent.app import load_saved_report, rerun_target_from_report
    from code_review_agent.reporter import build_report, save_report

    data = build_report(target=str(tmp_path / "proj"), provider="ollama", model="x")
    written = save_report(data, output_dir=str(tmp_path / "reports"), fmt="archive")
    md_path = next(p for p in written if p.endswith(".md"))
    status, markdown, rows, json_text, saved = load_saved_report(md_path)
    assert "Opened" in status
    assert "# Code Review Report" in markdown
    assert saved == md_path
    assert json_text
    assert rerun_target_from_report(md_path).endswith("proj")
    empty_status, empty_md, *_ = load_saved_report("")
    assert "Pick a saved report" in empty_status
    assert "No report selected" in empty_md


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_app_http_smoke():
    pytest.importorskip("gradio")
    from code_review_agent.app import _CSS, _theme, build_ui

    demo = build_ui()
    port = _free_port()
    kwargs = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "prevent_thread_lock": True,
        "inbrowser": False,
        "show_error": True,
    }
    try:
        try:
            demo.launch(theme=_theme(), css=_CSS, **kwargs)
        except TypeError:
            demo.launch(**kwargs)
        url = getattr(demo, "local_url", None) or f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
        assert status == 200
        lowered = body.lower()
        assert "quality triage" in lowered or "gradio" in lowered
    finally:
        demo.close()
