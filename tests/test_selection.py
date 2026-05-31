"""Offline tests for the analysis-selection layer."""

from __future__ import annotations

from code_review_agent import selection
from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI


class TestResolveFamilies:
    def test_empty_means_all(self):
        assert selection.resolve_families([]) == selection.ALL_FAMILIES
        assert selection.resolve_families(None) == selection.ALL_FAMILIES

    def test_subset_preserves_canonical_order(self):
        assert selection.resolve_families(["structural", "ml"]) == ["ml", "structural"]

    def test_unknown_only_falls_back_to_all(self):
        assert selection.resolve_families(["bogus"]) == selection.ALL_FAMILIES

    def test_skipped_is_complement(self):
        fams = selection.resolve_families(["ml"])
        skipped = selection.skipped_families(fams)
        assert "ml" not in skipped
        assert "code" in skipped and "td" in skipped


class TestTdCategories:
    def test_default_is_general(self):
        cats, unknown = selection.resolve_td_categories([])
        assert cats == ["general"] and unknown == []

    def test_valid_and_unknown_split(self):
        cats, unknown = selection.resolve_td_categories(["security", "nope", "design"])
        assert cats == ["security", "design"]
        assert unknown == ["nope"]

    def test_dedup(self):
        cats, _ = selection.resolve_td_categories(["security", "security"])
        assert cats == ["security"]


class TestToolRestriction:
    def test_allowed_tool_names_for_ml(self):
        allowed = selection.allowed_tool_names(["ml"])
        assert "detect_ml_smells" in allowed
        # navigation tools are always allowed
        assert "list_python_files" in allowed and "read_file" in allowed
        # python smells should NOT be allowed
        assert "detect_python_smells" not in allowed

    def test_td_family_unlocks_td_tools(self):
        allowed = selection.allowed_tool_names(["td"])
        assert "classify_technical_debt" in allowed
        assert "classify_technical_debt_all" in allowed
        assert "detect_ml_smells" not in allowed

    def test_filter_openai_tools_matches_allowed(self):
        defs = selection.filter_openai_tools(TOOL_DEFINITIONS_OPENAI, ["code-intel"])
        names = {d["function"]["name"] for d in defs}
        assert "analyze_code_intelligence" in names
        assert "detect_ml_smells" not in names
        # never exceeds the allowed set
        assert names <= selection.allowed_tool_names(["code-intel"])


class TestAnalysisType:
    def test_single_family(self):
        assert selection.python_analysis_type(["code"]) == "code"
        assert selection.python_analysis_type(["structural"]) == "structural"

    def test_all_three(self):
        assert selection.python_analysis_type(["code", "architectural", "structural"]) == "all"

    def test_none_when_no_python_family(self):
        assert selection.python_analysis_type(["ml", "td"]) is None

    def test_two_of_three_falls_back_to_all(self):
        assert selection.python_analysis_type(["code", "structural"]) == "all"


class TestSelectionNote:
    def test_note_lists_selected_and_skipped(self):
        note = selection.selection_note(["ml"], ["general"])
        assert "ml" in note
        assert "Skipped" in note
        assert "code" in note  # listed as skipped

    def test_note_includes_fix_instructions_when_requested(self):
        note = selection.selection_note(["ml"], ["general"], include_fix_instructions=True)
        assert "[[[FIX" in note

    def test_note_mentions_td_categories(self):
        note = selection.selection_note(["td"], ["security", "design"])
        assert "security" in note and "design" in note
