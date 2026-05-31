"""
Refactoring suggestions + safe, user-gated fix application.

The agent is asked to emit machine-applicable fix blocks in a delimiter-based
format (so they never collide with normal Markdown code fences):

    [[[FIX file="path/to/file.py" lines=10-12 desc="Add random seed"]]]
    --- ORIGINAL
    <the exact current lines>
    --- FIXED
    <the replacement lines>
    [[[/FIX]]]

This module parses those blocks, renders unified diffs, and applies them under
strict safety rules:

  * Default is SUGGEST-ONLY (``dry_run=True``): no files are ever written.
  * Applying requires BOTH ``dry_run=False`` AND ``confirm=True``.
  * Writes are confined to ``project_root`` — paths resolving outside it are
    refused (never write outside the target project).
  * The current file contents at the target lines must match the block's
    ORIGINAL text, otherwise the fix is skipped (no blind overwrites).
  * A ``.bak`` backup is written before modifying a file.

Nothing here imports the LLM or any heavy dependency, so it is fully testable
offline.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIX_BLOCK_INSTRUCTIONS = (
    "## Suggested Fixes (machine-applicable)\n"
    "After the report, for the most important issues that have a concrete code "
    "fix, emit one or more FIX blocks in EXACTLY this format (do not wrap them in "
    "Markdown code fences):\n"
    "\n"
    '[[[FIX file="<relative path within the project>" lines=<start>-<end> desc="<short description>"]]]\n'
    "--- ORIGINAL\n"
    "<the exact current source lines being replaced>\n"
    "--- FIXED\n"
    "<the corrected source lines>\n"
    "[[[/FIX]]]\n"
    "\n"
    "Rules for FIX blocks: the ORIGINAL section MUST be the verbatim current "
    "lines (so they can be matched safely), `lines` are 1-based inclusive, and "
    "the file path must be relative to the project root. Only suggest fixes you "
    "are confident about. If a finding has no safe automatic fix, describe it in "
    "the report instead of emitting a FIX block."
)

_FIX_OPEN = re.compile(
    r"\[\[\[FIX\b(?P<attrs>[^\]]*)\]\]\](?P<body>.*?)\[\[\[/FIX\]\]\]",
    re.DOTALL,
)
# Attributes may be quoted ("...") or bare (lines=2-12). Capture both.
_ATTR = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))')


@dataclass
class FixSuggestion:
    file: str
    start_line: int
    end_line: int
    original: str
    replacement: str
    description: str = ""
    finding_id: str | None = None
    source: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "original": self.original,
            "replacement": self.replacement,
            "description": self.description,
            "finding_id": self.finding_id,
            "source": self.source,
        }


def _parse_lines_attr(value: str) -> tuple[int, int]:
    value = (value or "").strip()
    m = re.match(r"(\d+)\s*-\s*(\d+)", value)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"(\d+)", value)
    if m:
        n = int(m.group(1))
        return n, n
    return 0, 0


def _split_original_fixed(body: str) -> tuple[str, str]:
    """Split a FIX body into (original, fixed) on the ``--- FIXED`` marker."""
    # Tolerate optional leading "--- ORIGINAL" marker.
    original_marker = re.search(r"---\s*ORIGINAL\s*\n", body)
    start = original_marker.end() if original_marker else 0
    fixed_marker = re.search(r"\n?---\s*FIXED\s*\n", body[start:])
    if not fixed_marker:
        return body[start:].strip("\n"), ""
    original = body[start : start + fixed_marker.start()]
    fixed = body[start + fixed_marker.end() :]
    return original.strip("\n"), fixed.strip("\n")


def parse_fix_blocks(text: str) -> list[FixSuggestion]:
    """Extract FixSuggestion objects from agent output text."""
    suggestions: list[FixSuggestion] = []
    if not text:
        return suggestions
    for match in _FIX_OPEN.finditer(text):
        attrs = {k: (q or u) for k, q, u in _ATTR.findall(match.group("attrs"))}
        start, end = _parse_lines_attr(attrs.get("lines", ""))
        original, fixed = _split_original_fixed(match.group("body"))
        file_attr = attrs.get("file", "").strip()
        if not file_attr:
            continue
        suggestions.append(
            FixSuggestion(
                file=file_attr,
                start_line=start,
                end_line=end or start,
                original=original,
                replacement=fixed,
                description=attrs.get("desc", ""),
                finding_id=attrs.get("id") or None,
                source="agent",
            )
        )
    return suggestions


def make_unified_diff(file: str, original: str, replacement: str) -> str:
    """Render a unified diff between original and replacement text."""
    orig_lines = original.splitlines()
    new_lines = replacement.splitlines()
    diff = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=f"a/{file}",
        tofile=f"b/{file}",
        lineterm="",
    )
    return "\n".join(diff)


def _resolve_within_root(file: str, project_root: str) -> tuple[Path | None, str | None]:
    """Resolve ``file`` and ensure it stays inside ``project_root``."""
    root = Path(project_root).resolve()
    candidate = Path(file)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, "path resolves outside the project root"
    return resolved, None


@dataclass
class FixOutcome:
    diffs: list[dict] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "skipped": self.skipped,
            "diffs": self.diffs,
            "counts": {
                "applied": len(self.applied),
                "skipped": len(self.skipped),
                "diffs": len(self.diffs),
            },
        }


def apply_fixes(
    suggestions: list[FixSuggestion],
    project_root: str,
    dry_run: bool = True,
    confirm: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Preview (and optionally apply) fix suggestions under strict safety rules.

    Returns a structured dict (see :class:`FixOutcome`). With ``dry_run=True``
    (the default) NO files are written. Writing also requires ``confirm=True``.
    """
    outcome = FixOutcome()

    for idx, s in enumerate(suggestions):
        tag = s.finding_id or f"fix-{idx + 1:03d}"
        resolved, err = _resolve_within_root(s.file, project_root)
        if err:
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": err})
            continue
        if not resolved.exists():
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": "file not found"})
            continue

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": f"read failed: {exc}"})
            continue

        diff_text = make_unified_diff(s.file, s.original, s.replacement)
        outcome.diffs.append(
            {
                "fix": tag,
                "file": s.file,
                "lines": f"{s.start_line}-{s.end_line}",
                "description": s.description,
                "diff": diff_text,
            }
        )

        file_lines = content.splitlines()
        start, end = s.start_line, s.end_line
        match_ok = False
        if 1 <= start <= end <= len(file_lines):
            current = "\n".join(file_lines[start - 1 : end])
            match_ok = current.strip() == s.original.strip()

        # ---- Safety gating: never write unless explicitly told to. ----
        if dry_run:
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": "dry-run (no write)"})
            continue
        if not confirm:
            outcome.skipped.append(
                {
                    "fix": tag,
                    "file": s.file,
                    "reason": "apply requires explicit confirmation",
                }
            )
            continue
        if not s.original.strip():
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": "empty ORIGINAL block"})
            continue
        if not match_ok:
            outcome.skipped.append(
                {
                    "fix": tag,
                    "file": s.file,
                    "reason": "ORIGINAL block does not match current file contents",
                }
            )
            continue

        new_lines = file_lines[: start - 1] + s.replacement.splitlines() + file_lines[end:]
        trailing_nl = "\n" if content.endswith("\n") else ""
        new_content = "\n".join(new_lines) + trailing_nl
        try:
            if backup:
                backup_path = resolved.with_name(resolved.name + ".bak")
                backup_path.write_text(content, encoding="utf-8")
            resolved.write_text(new_content, encoding="utf-8")
            outcome.applied.append(
                {
                    "fix": tag,
                    "file": s.file,
                    "lines": f"{start}-{end}",
                    "backup": str(resolved.name + ".bak") if backup else None,
                }
            )
        except Exception as exc:
            outcome.skipped.append({"fix": tag, "file": s.file, "reason": f"write failed: {exc}"})

    return outcome.to_dict()


def render_fixes_markdown(suggestions: list[FixSuggestion], outcome: dict | None = None) -> str:
    """Render a Markdown 'Suggested Fixes' section from parsed suggestions."""
    if not suggestions:
        return ""
    lines: list[str] = ["## Suggested Fixes", ""]
    lines.append(f"_{len(suggestions)} fix suggestion(s) parsed from the review._")
    lines.append("")
    diffs_by_idx = {}
    if outcome:
        for i, d in enumerate(outcome.get("diffs", [])):
            diffs_by_idx[i] = d.get("diff", "")
    for i, s in enumerate(suggestions):
        title = s.description or f"{s.file}:{s.start_line}-{s.end_line}"
        lines.append(f"### Fix {i + 1}: {title}")
        lines.append("")
        lines.append(f"**File:** `{s.file}`  ")
        lines.append(f"**Lines:** {s.start_line}-{s.end_line}  ")
        lines.append("")
        diff_text = diffs_by_idx.get(i) or make_unified_diff(s.file, s.original, s.replacement)
        lines.append("```diff")
        lines.append(diff_text)
        lines.append("```")
        lines.append("")
    if outcome:
        counts = outcome.get("counts", {})
        lines.append(
            f"_Apply summary: {counts.get('applied', 0)} applied, "
            f"{counts.get('skipped', 0)} skipped._"
        )
        lines.append("")
    return "\n".join(lines)
