"""Offline tests for fix parsing, diffing, and SAFE gated application."""

from __future__ import annotations

from code_review_agent import fixes
from code_review_agent.fixes import FixSuggestion, apply_fixes, parse_fix_blocks


SAMPLE_TEXT = """Here is my review.

[[[FIX file="mod.py" lines=2-2 desc="Add random seed"]]]
--- ORIGINAL
model = train()
--- FIXED
np.random.seed(42)
model = train()
[[[/FIX]]]

Done.
"""


class TestParse:
    def test_parses_single_block(self):
        suggestions = parse_fix_blocks(SAMPLE_TEXT)
        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.file == "mod.py"
        assert s.start_line == 2 and s.end_line == 2
        assert s.original == "model = train()"
        assert "np.random.seed(42)" in s.replacement
        assert s.description == "Add random seed"

    def test_no_blocks_returns_empty(self):
        assert parse_fix_blocks("no fixes here") == []

    def test_handles_multiple_blocks(self):
        text = SAMPLE_TEXT + SAMPLE_TEXT.replace("mod.py", "other.py")
        assert len(parse_fix_blocks(text)) == 2


class TestDiff:
    def test_unified_diff_has_markers(self):
        d = fixes.make_unified_diff("mod.py", "a = 1", "a = 2")
        assert "-a = 1" in d and "+a = 2" in d


def _mk_project(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("import numpy as np\nmodel = train()\n", encoding="utf-8")
    return f


class TestApplySafety:
    def test_dry_run_never_writes(self, tmp_path):
        f = _mk_project(tmp_path)
        before = f.read_text(encoding="utf-8")
        suggestions = parse_fix_blocks(SAMPLE_TEXT)
        out = apply_fixes(suggestions, str(tmp_path), dry_run=True, confirm=False)
        assert f.read_text(encoding="utf-8") == before          # unchanged
        assert out["counts"]["applied"] == 0
        assert out["counts"]["diffs"] == 1                       # preview produced
        assert any("dry-run" in s["reason"] for s in out["skipped"])

    def test_apply_without_confirm_is_refused(self, tmp_path):
        f = _mk_project(tmp_path)
        before = f.read_text(encoding="utf-8")
        suggestions = parse_fix_blocks(SAMPLE_TEXT)
        out = apply_fixes(suggestions, str(tmp_path), dry_run=False, confirm=False)
        assert f.read_text(encoding="utf-8") == before          # still unchanged
        assert out["counts"]["applied"] == 0
        assert any("confirmation" in s["reason"] for s in out["skipped"])

    def test_apply_with_confirm_writes_and_backs_up(self, tmp_path):
        f = _mk_project(tmp_path)
        suggestions = parse_fix_blocks(SAMPLE_TEXT)
        out = apply_fixes(suggestions, str(tmp_path), dry_run=False, confirm=True)
        assert out["counts"]["applied"] == 1
        new = f.read_text(encoding="utf-8")
        assert "np.random.seed(42)" in new
        assert (tmp_path / "mod.py.bak").exists()               # backup made

    def test_refuses_path_outside_project_root(self, tmp_path):
        outside = FixSuggestion(
            file="../escape.py", start_line=1, end_line=1,
            original="x = 1", replacement="x = 2",
        )
        out = apply_fixes([outside], str(tmp_path), dry_run=False, confirm=True)
        assert out["counts"]["applied"] == 0
        assert any("outside" in s["reason"] for s in out["skipped"])

    def test_mismatched_original_is_skipped(self, tmp_path):
        f = _mk_project(tmp_path)
        bad = FixSuggestion(
            file="mod.py", start_line=2, end_line=2,
            original="this is not the real line", replacement="model = retrain()",
        )
        out = apply_fixes([bad], str(tmp_path), dry_run=False, confirm=True)
        assert out["counts"]["applied"] == 0
        assert any("does not match" in s["reason"] for s in out["skipped"])
        assert "model = train()" in f.read_text(encoding="utf-8")


class TestRender:
    def test_render_markdown_section(self):
        suggestions = parse_fix_blocks(SAMPLE_TEXT)
        md = fixes.render_fixes_markdown(suggestions)
        assert "## Suggested Fixes" in md
        assert "```diff" in md
