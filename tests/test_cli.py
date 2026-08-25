"""CLI tests using Click's CliRunner (no live LLM)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from code_review_agent.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "review" in result.output
    assert "CLI-first" in result.output
    assert "code-review app" in result.output


def test_review_help_lists_flags():
    runner = CliRunner()
    result = runner.invoke(main, ["review", "--help"])
    assert result.exit_code == 0
    for flag in ("--no-llm", "--fail-on", "--format", "--agentic", "--model"):
        assert flag in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_show_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  model: ollama/test\n")
    runner = CliRunner()
    result = runner.invoke(main, ["show-config"])
    assert result.exit_code == 0
    assert "ollama/test" in result.output


def test_list_tools():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tools"])
    assert result.exit_code == 0
    assert "detect_ml_smells" in result.output


def test_models_lists_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["models"])
    assert result.exit_code == 0
    assert "frontier" in result.output
    assert "local" in result.output
    assert "test" in result.output


def test_hidden_ollama_models_alias():
    runner = CliRunner()
    result = runner.invoke(main, ["ollama-models", "--help"])
    assert result.exit_code == 0


def test_review_missing_path():
    runner = CliRunner()
    result = runner.invoke(main, ["review", "/definitely/missing/path"])
    assert result.exit_code == EXIT_ERROR


def test_review_no_llm(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["-q", "review", str(tmp_path), "--no-llm", "--format", "json"]
    )
    assert result.exit_code in (EXIT_OK, EXIT_FINDINGS)
    assert '"findings"' in result.output or result.exit_code == EXIT_FINDINGS


def test_review_fail_on_none(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            str(tmp_path),
            "--no-llm",
            "--format",
            "json",
            "--fail-on",
            "none",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert '"target"' in result.output


def test_run_tool_list_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run-tool", "list-files", str(tmp_path)])
    assert result.exit_code == 0
    assert "a.py" in result.output


def test_run_tool_code_intel(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-tool",
            "code-intel",
            str(tmp_path),
            "--top-n",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert "foo" in result.output or "total_functions" in result.output


def test_review_ci_no_llm(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "review.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--ci",
            "review",
            str(tmp_path),
            "--no-llm",
            "--format",
            "json",
            "--fail-on",
            "none",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == EXIT_OK
    payload = json.loads(out.read_text())
    assert "findings" in payload
    assert "health_score" in payload


def test_run_tool_list_files_missing_path():
    runner = CliRunner()
    result = runner.invoke(main, ["run-tool", "list-files", "/no/such/qt-dir"])
    assert result.exit_code == EXIT_ERROR


def test_run_tool_read_file_missing_exits():
    runner = CliRunner()
    result = runner.invoke(main, ["run-tool", "read-file", "/no/such/file.py"])
    assert result.exit_code != 0


def test_analyze_file_no_llm(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("def bar(x):\n    return x\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["-q", "analyze-file", str(f), "--no-llm", "--format", "json"]
    )
    assert result.exit_code in (EXIT_OK, EXIT_FINDINGS)
    assert '"findings"' in result.output or result.exit_code == EXIT_FINDINGS


def test_review_sarif_output(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "review.sarif"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            str(tmp_path),
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "sarif",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["version"] == "2.1.0"


def test_review_html_output(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "review.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            str(tmp_path),
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "html",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "Quality Triage" in out.read_text()


def test_review_autosaves_markdown_archive(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["-q", "review", str(tmp_path), "--no-llm", "--fail-on", "none"],
    )
    assert result.exit_code == EXIT_OK, result.output
    reports = list((tmp_path / "reports").glob("review-*.md"))
    assert reports
    text = reports[0].read_text(encoding="utf-8")
    assert "# Code Review Report" in text
    assert (tmp_path / "reports" / reports[0].with_suffix(".json").name).is_file()


def test_review_json_includes_health(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            str(tmp_path),
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == EXIT_OK
    payload, _ = json.JSONDecoder().raw_decode(result.output.lstrip())
    assert "health_score" in payload


def test_review_github_passes_issue_texts(tmp_path, monkeypatch):
    from code_review_agent.github_utils import ClonedRepo, GithubIssue
    from code_review_agent.pipeline import PipelineResult
    from code_review_agent.reporter import build_report

    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n")
    captured: dict = {}
    cloned = ClonedRepo(
        url="https://github.com/acme/demo",
        local_path=str(tmp_path),
        repo_name="acme/demo",
        branch="main",
        commit_sha="abc12345deadbeef",
        is_temp=False,
        issues=[GithubIssue(number=7, title="Fix hacks", body="remove deprecated")],
    )

    def fake_resolve(target, **kwargs):
        captured["fetch_issues"] = kwargs.get("fetch_issues")
        captured["issue_limit"] = kwargs.get("issue_limit")
        return str(tmp_path), cloned

    def fake_exec(path, cfg, **kwargs):
        captured["issue_texts"] = kwargs.get("issue_texts")
        report = build_report(target=path, provider="ollama", model="x")
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
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            "https://github.com/acme/demo",
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert captured["fetch_issues"] is True
    assert captured["issue_limit"] == 10
    assert captured["issue_texts"]
    assert captured["issue_texts"][0].startswith("#7 Fix hacks")


def test_app_help():
    runner = CliRunner()
    result = runner.invoke(main, ["app", "--help"])
    assert result.exit_code == 0
    assert "Gradio" in result.output


def test_run_tool_classify_td_missing_text():
    runner = CliRunner()
    result = runner.invoke(main, ["run-tool", "classify-td"])
    assert result.exit_code == EXIT_ERROR
