"""Live GitHub clone + pipeline (needs network and git).

Skip with QUALITY_TRIAGE_SKIP_E2E=1. Clone failures skip so offline CI stays green.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from code_review_agent.app import run_review, run_tool
from code_review_agent.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main
from code_review_agent.github_utils import cleanup_repo, resolve_review_target

# Tiny public Python project; /tree/main/src exercises branch + subpath parsing.
GITHUB_SRC = "https://github.com/pypa/sampleproject/tree/main/src"
PALLETS_ISSUE_SNIPPETS = (
    Path(__file__).resolve().parent / "fixtures" / "pallets_issue_snippets.txt"
)
FLASK_REPO = "pallets/flask"
CLICK_REPO = "pallets/click"


def _require_e2e() -> None:
    if os.environ.get("QUALITY_TRIAGE_SKIP_E2E") == "1":
        pytest.skip("QUALITY_TRIAGE_SKIP_E2E=1")


def _skip_if_clone_failed(result) -> None:
    blob = f"{result.output or ''}{getattr(result, 'stderr', '') or ''}".lower()
    if result.exit_code == EXIT_ERROR and (
        "clone failed" in blob or "git clone" in blob or "not a github" in blob
    ):
        pytest.skip(f"GitHub clone unavailable: {blob[:300]}")


@pytest.mark.e2e
def test_clone_github_subpath_and_cleanup():
    _require_e2e()
    try:
        path, cloned = resolve_review_target(GITHUB_SRC)
    except Exception as exc:
        pytest.skip(f"GitHub clone unavailable: {exc}")
    try:
        assert cloned is not None
        assert cloned.subpath == "src"
        review = Path(path)
        assert review.is_dir()
        py_files = list(review.rglob("*.py"))
        assert py_files, "expected Python files under GitHub subpath"
        assert any(p.name == "simple.py" for p in py_files)
        assert isinstance(cloned.issues, list)
        assert len(cloned.issues) <= 10
    finally:
        if cloned:
            cleanup_repo(cloned)
            assert not Path(cloned.local_path).exists()


@pytest.mark.e2e
def test_cli_review_github_json(tmp_path, monkeypatch):
    _require_e2e()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "github-review.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            GITHUB_SRC,
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "json",
            "--output",
            str(out),
        ],
    )
    _skip_if_clone_failed(result)
    assert result.exit_code in (EXIT_OK, EXIT_FINDINGS)
    assert out.is_file(), result.output
    payload = json.loads(out.read_text())
    assert payload.get("files_analyzed", 0) >= 1
    assert "health_score" in payload
    assert "findings" in payload
    assert "sampleproject" in payload.get("target", "") or "src" in payload.get(
        "target", ""
    )


@pytest.mark.e2e
def test_cli_review_github_html(tmp_path, monkeypatch):
    _require_e2e()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "github-review.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-q",
            "review",
            GITHUB_SRC,
            "--no-llm",
            "--fail-on",
            "none",
            "--format",
            "html",
            "--output",
            str(out),
        ],
    )
    _skip_if_clone_failed(result)
    assert result.exit_code in (EXIT_OK, EXIT_FINDINGS)
    html = out.read_text()
    assert "Quality Triage" in html
    assert "inspection slip" in html


@pytest.mark.e2e
def test_cli_run_tool_github_list_files():
    _require_e2e()
    runner = CliRunner()
    result = runner.invoke(main, ["run-tool", "list-files", GITHUB_SRC])
    _skip_if_clone_failed(result)
    assert result.exit_code == EXIT_OK
    assert "simple.py" in result.output or ".py" in result.output


@pytest.mark.e2e
def test_cli_run_tool_github_code_intel():
    _require_e2e()
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-tool",
            "code-intel",
            GITHUB_SRC,
            "--top-n",
            "5",
            "--format",
            "json",
        ],
    )
    _skip_if_clone_failed(result)
    assert result.exit_code == EXIT_OK
    assert "simple" in result.output or "total_functions" in result.output


@pytest.mark.e2e
def test_app_run_review_github():
    _require_e2e()
    chunks = list(run_review(GITHUB_SRC, None, "local", True, "", None))
    assert chunks
    status, report_md, rows, json_text, saved, *rest = chunks[-1]
    if status.startswith("Failed:"):
        pytest.skip(f"GitHub review unavailable: {status}")
    assert "# Code Review Report" in report_md
    payload = json.loads(json_text)
    assert payload.get("files_analyzed", 0) >= 1
    assert "health_score" in payload
    assert isinstance(rows, list)
    assert saved
    assert ".md" in saved


@pytest.mark.e2e
def test_app_run_tool_github_list_files():
    _require_e2e()
    raw = run_tool("list-files", GITHUB_SRC, None, "", None)
    payload = json.loads(raw)
    if "error" in payload:
        pytest.skip(f"GitHub tool unavailable: {payload['error']}")
    assert payload.get("total_files", 0) >= 1
    names = " ".join(str(f) for f in payload.get("files", payload.get("paths", [])))
    blob = json.dumps(payload)
    assert "simple.py" in blob or names


def _load_pallets_snippets() -> list[str]:
    lines = [
        line.strip()
        for line in PALLETS_ISSUE_SNIPPETS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 8
    return lines


def _gh_issue_snippets(repo: str, limit: int = 8) -> list[str]:
    if shutil.which("gh") is None:
        pytest.skip("gh CLI is not installed")
    proc = subprocess.run(
        [
            "gh",
            "search",
            "issues",
            "--repo",
            repo,
            "--limit",
            str(limit),
            "--json",
            "number,title,body",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.returncode != 0:
        pytest.skip(f"gh search failed for {repo}: {proc.stderr[:300]}")
    try:
        issues = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.skip(f"gh returned non-JSON for {repo}: {exc}")
    snippets: list[str] = []
    for issue in issues:
        title = (issue.get("title") or "").strip()
        if title.lower() == "ai junk":
            continue
        body = " ".join((issue.get("body") or "").split())
        snippets.append(f"{repo}#{issue.get('number')} {title}. {body}"[:800])
    if not snippets:
        pytest.skip(f"no usable issues from {repo}")
    return snippets


@pytest.mark.e2e
def test_cli_classify_td_pallets_issue_fixtures(tmp_path, monkeypatch):
    _require_e2e()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "pallets-td.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-tool",
            "classify-td",
            "--from-file",
            str(PALLETS_ISSUE_SNIPPETS),
            "--format",
            "json",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(out.read_text())
    if "error" in payload:
        pytest.skip(f"TD classifier unavailable: {payload['error']}")
    preds = payload.get("predictions", [])
    assert len(preds) == len(_load_pallets_snippets())
    assert all("predicted_class" in p or "error" in p for p in preds)
    assert any(p.get("predicted_class") == 1 for p in preds)


@pytest.mark.e2e
def test_classify_td_live_flask_and_click_issues():
    _require_e2e()
    from code_review_agent.tools import classify_technical_debt

    snippets = _gh_issue_snippets(FLASK_REPO) + _gh_issue_snippets(CLICK_REPO)
    result = classify_technical_debt(snippets)
    if "error" in result:
        pytest.skip(f"TD classifier unavailable: {result['error']}")
    preds = result.get("predictions", [])
    assert len(preds) == len(snippets)
    assert all("predicted_class" in p or "error" in p for p in preds)
    # App helper used by Gradio Tools tab
    raw = run_tool("classify-td", "", None, "\n".join(snippets[:4]), None)
    app_payload = json.loads(raw)
    if "error" in app_payload:
        pytest.skip(f"Gradio classify-td unavailable: {app_payload['error']}")
    assert len(app_payload.get("predictions", [])) == 4


@pytest.mark.e2e
def test_fetch_recent_open_issues_pallets_click():
    _require_e2e()
    from code_review_agent.github_utils import fetch_recent_open_issues

    issues = fetch_recent_open_issues("pallets/click", limit=10)
    if not issues:
        pytest.skip("GitHub issues API unavailable or pallets/click has no open issues")
    assert len(issues) <= 10
    assert all(issue.title for issue in issues)
    assert all(issue.number for issue in issues)
    snippets = [issue.snippet() for issue in issues]
    from code_review_agent.tools import classify_technical_debt

    result = classify_technical_debt(snippets)
    if "error" in result:
        pytest.skip(f"TD classifier unavailable: {result['error']}")
    assert len(result.get("predictions", [])) == len(snippets)
