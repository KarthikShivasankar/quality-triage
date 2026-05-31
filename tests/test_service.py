"""Offline tests for the shared review orchestration service."""

from __future__ import annotations

from code_review_agent import selection, service
from code_review_agent.config import get_config


class TestResolveSelection:
    def test_defaults_from_config(self):
        cfg = get_config()
        sel = service.resolve_selection(cfg, [], [])
        assert set(sel["families"]) == set(selection.ALL_FAMILIES)
        assert sel["td_categories"] == ["general"]

    def test_cli_overrides_config(self):
        cfg = get_config()
        sel = service.resolve_selection(cfg, ["ml", "structural"], ["security", "bogus"])
        assert sel["families"] == ["ml", "structural"]
        assert sel["td_categories"] == ["security"]
        assert sel["unknown_td_categories"] == ["bogus"]
        assert "code" in sel["skipped"]


class TestBuildAgentRestriction:
    def test_tool_schema_is_restricted(self):
        cfg = get_config()
        agent = service.build_agent(cfg, "ollama", None, ["ml"])
        assert agent.allowed_tool_names == selection.allowed_tool_names(["ml"])
        names = {t["function"]["name"] for t in agent.tools}
        assert names <= selection.allowed_tool_names(["ml"])
        assert "detect_ml_smells" in names
        assert "detect_python_smells" not in names

    def test_no_families_means_full_toolset(self):
        cfg = get_config()
        agent = service.build_agent(cfg, "ollama", None, None)
        assert agent.allowed_tool_names is None


class _FakeAgent:
    def __init__(self, text):
        self._text = text

    def review_text(self, path, extra_context=""):
        self._ctx = extra_context
        return self._text


class TestRunReview:
    def test_parses_fixes_and_selection(self, monkeypatch):
        cfg = get_config()
        text = (
            "Report body.\n"
            '[[[FIX file="mod.py" lines=1-1 desc="x"]]]\n'
            "--- ORIGINAL\na = 1\n--- FIXED\na = 2\n[[[/FIX]]]\n"
        )
        monkeypatch.setattr(service, "build_agent", lambda *a, **k: _FakeAgent(text))
        result = service.run_review(
            cfg, "proj", checks=["ml"], td_categories=["security"], suggest_fixes=True
        )
        assert result["selection"]["families"] == ["ml"]
        assert len(result["fixes"]) == 1
        assert result["fixes"][0]["file"] == "mod.py"

    def test_no_fixes_when_not_requested(self, monkeypatch):
        cfg = get_config()
        monkeypatch.setattr(service, "build_agent", lambda *a, **k: _FakeAgent("no fix blocks"))
        result = service.run_review(cfg, "proj", suggest_fixes=False)
        assert result["fixes"] == []
