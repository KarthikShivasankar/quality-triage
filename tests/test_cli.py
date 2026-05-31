"""
Tests for code_review_agent.cli via click's CliRunner — fully offline.

The LLM agent is mocked for ask/review so no network calls happen. Detector
commands run the REAL (installed) analyzers on temp files.
"""

from __future__ import annotations

import importlib.util

import pytest
from click.testing import CliRunner

from code_review_agent.cli import main


HAS_CQA = importlib.util.find_spec("code_quality_analyzer") is not None
HAS_ML = importlib.util.find_spec("ml_code_smell_detector") is not None


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "mod.py").write_text(
        "import numpy as np\n\n\ndef f(a, b, c, d, e, f, g):\n"
        + "    x = 1\n" * 30
        + "    return x\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Informational commands
# ---------------------------------------------------------------------------

class TestInfoCommands:
    def test_show_config_includes_openai(self, runner):
        result = runner.invoke(main, ["show-config"])
        assert result.exit_code == 0, result.output
        assert "openai" in result.output
        assert "ollama" in result.output
        assert "anthropic" in result.output

    def test_list_tools(self, runner):
        result = runner.invoke(main, ["list-tools"])
        assert result.exit_code == 0, result.output
        assert "detect_ml_smells" in result.output
        assert "classify_technical_debt" in result.output

    def test_providers(self, runner):
        result = runner.invoke(main, ["providers"])
        assert result.exit_code == 0, result.output
        assert "ollama" in result.output
        assert "openai" in result.output
        assert "anthropic" in result.output

    def test_doctor_runs_without_crash(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "doctor" in result.output.lower()
        # Detector availability is reported.
        assert "code_quality_analyzer" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "review" in result.output
        assert "providers" in result.output
        assert "doctor" in result.output


# ---------------------------------------------------------------------------
# run-tool: pure-Python tools (no third-party deps)
# ---------------------------------------------------------------------------

class TestRunToolPure:
    def test_list_files(self, runner, proj):
        result = runner.invoke(main, ["run-tool", "list-files", str(proj)])
        assert result.exit_code == 0, result.output
        assert "mod.py" in result.output

    def test_read_file(self, runner, proj):
        result = runner.invoke(main, ["run-tool", "read-file", str(proj / "mod.py")])
        assert result.exit_code == 0, result.output
        assert "numpy" in result.output

    def test_code_intel(self, runner, proj):
        result = runner.invoke(main, ["run-tool", "code-intel", str(proj)])
        assert result.exit_code == 0, result.output
        assert "Code Intelligence" in result.output


# ---------------------------------------------------------------------------
# run-tool: real detectors (import-guarded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CQA, reason="code_quality_analyzer not installed")
class TestRunToolPythonSmells:
    def test_code_smells_json(self, runner, proj):
        result = runner.invoke(
            main, ["run-tool", "python-smells", str(proj), "--type", "code", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        assert "code_smells" in result.output

    def test_structural_table(self, runner, proj):
        result = runner.invoke(
            main, ["run-tool", "python-smells", str(proj), "--type", "structural"]
        )
        assert result.exit_code == 0, result.output


@pytest.mark.skipif(not HAS_ML, reason="ml_code_smell_detector not installed")
class TestRunToolMlSmells:
    def test_ml_smells_json(self, runner, proj):
        result = runner.invoke(
            main, ["run-tool", "ml-smells", str(proj), "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        assert "ml_smells" in result.output or "summary" in result.output


# ---------------------------------------------------------------------------
# Agent-driven commands with a mocked agent (no network)
# ---------------------------------------------------------------------------

class _FakeAgent:
    def review(self, path, extra_context=""):
        yield "MOCK REVIEW for "
        yield path

    def ask(self, question):
        yield "MOCK ANSWER: "
        yield question


@pytest.fixture
def mock_agent(monkeypatch):
    import code_review_agent.cli as cli_mod

    def _fake_make_agent(cfg, provider_override=None, model_override=None, base_url=None,
                         api_key=None, families=None):
        return _FakeAgent()

    monkeypatch.setattr(cli_mod, "_make_agent", _fake_make_agent)


class TestAgentCommands:
    def test_ask_mocked(self, runner, mock_agent):
        result = runner.invoke(main, ["ask", "what is tech debt?"])
        assert result.exit_code == 0, result.output
        assert "MOCK ANSWER" in result.output

    def test_review_local_path_mocked(self, runner, mock_agent, proj):
        result = runner.invoke(main, ["review", str(proj)])
        assert result.exit_code == 0, result.output
        assert "MOCK REVIEW" in result.output

    def test_review_nonexistent_path(self, runner, mock_agent):
        result = runner.invoke(main, ["review", "/no/such/path/xyz"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Selection + fix flow (mocked agent, fully offline)
# ---------------------------------------------------------------------------

_FIX_TEXT = (
    "Review.\n"
    '[[[FIX file="mod.py" lines=1-1 desc="Annotate import"]]]\n'
    "--- ORIGINAL\nimport numpy as np\n--- FIXED\nimport numpy as np  # seeded\n[[[/FIX]]]\n"
)


class _FixAgent:
    def review(self, path, extra_context=""):
        self.extra_context = extra_context
        yield _FIX_TEXT


@pytest.fixture
def fix_agent(monkeypatch):
    import code_review_agent.cli as cli_mod
    holder = {}

    def _fake_make_agent(cfg, provider_override=None, model_override=None, base_url=None,
                         api_key=None, families=None):
        holder["families"] = families
        return _FixAgent()

    monkeypatch.setattr(cli_mod, "_make_agent", _fake_make_agent)
    return holder


class TestReviewSelection:
    def test_review_reports_selected_and_skipped(self, runner, mock_agent, proj):
        result = runner.invoke(main, ["review", str(proj), "--check", "ml", "--check", "structural"])
        assert result.exit_code == 0, result.output
        assert "ml, structural" in result.output
        assert "Skipped" in result.output

    def test_review_passes_families_to_agent(self, runner, fix_agent, proj):
        result = runner.invoke(main, ["review", str(proj), "--check", "ml"])
        assert result.exit_code == 0, result.output
        assert fix_agent["families"] == ["ml"]


class TestReviewFixes:
    def test_suggest_fixes_dry_run_does_not_write(self, runner, fix_agent, proj):
        before = (proj / "mod.py").read_text(encoding="utf-8")
        result = runner.invoke(main, ["review", str(proj), "--suggest-fixes", "--fix-dry-run"])
        assert result.exit_code == 0, result.output
        assert "Parsed 1 fix suggestion" in result.output
        assert (proj / "mod.py").read_text(encoding="utf-8") == before  # unchanged

    def test_apply_fixes_with_yes_writes(self, runner, fix_agent, proj):
        result = runner.invoke(main, ["review", str(proj), "--apply-fixes", "--yes"])
        assert result.exit_code == 0, result.output
        assert "# seeded" in (proj / "mod.py").read_text(encoding="utf-8")
        assert (proj / "mod.py.bak").exists()

    def test_review_json_output(self, runner, fix_agent, proj, tmp_path):
        out = tmp_path / "out.json"
        result = runner.invoke(
            main, ["review", str(proj), "--suggest-fixes", "--fix-dry-run",
                   "--format", "json", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        import json as _json
        data = _json.loads(out.read_text(encoding="utf-8"))
        assert data["selection"]["families"]
        assert data["report_markdown"]


class TestRunToolSeverity:
    def test_python_smells_min_severity(self, runner, proj):
        result = runner.invoke(
            main, ["run-tool", "python-smells", str(proj), "--type", "structural",
                   "--min-severity", "high", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
