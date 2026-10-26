"""Tests for the detector-verified refactoring harness."""

from __future__ import annotations

import sys
import textwrap

from click.testing import CliRunner

from code_review_agent.llm import CompletionResult
from code_review_agent.refactor import (
    CheckCommand,
    Harness,
    enclosing_unit,
    gaming_flags,
    interface_preserved,
    parse_reply,
    relocate,
    repetition_ratio,
    select_targets,
    splice,
)

SOURCE = textwrap.dedent(
    '''\
    """Module."""
    import os


    class Box:
        def size(self, a, b=1):
            if a:
                if b:
                    return a + b
            return 0


    def top(x):
        return x * 2
    '''
)

NESTED = "def nested(a, b):\n    if a:\n        if b:\n            return a + b\n    return 0\n"
FLAT = "def nested(a, b):\n    if a and b:\n        return a + b\n    return 0\n"


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_text(self, messages, tools=None):
        self.calls.append(messages)
        return CompletionResult(content=self.replies.pop(0), finish_reason="stop")


def nesting_detector(root):
    """Flags every `if` nested directly in another `if` as 'Deep Nesting'."""
    import ast

    out = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and any(
                isinstance(n, ast.If) for n in node.body
            ):
                out.append(
                    {
                        "tool": "python_smells",
                        "category": "Deep Nesting",
                        "file": path.relative_to(root).as_posix(),
                        "line": node.lineno,
                        "message": "nested if",
                    }
                )
    return out


def test_enclosing_unit_prefers_function():
    unit = enclosing_unit(SOURCE, 7, "m.py")
    assert (unit.name, unit.kind, unit.start, unit.end, unit.indent) == (
        "size",
        "function",
        6,
        10,
        4,
    )


def test_select_targets_skips_module_level_and_unfixable(tmp_path):
    (tmp_path / "m.py").write_text(SOURCE)
    findings = [
        {"category": "Deep Nesting", "file": "m.py", "line": 7},
        {"category": "Deep Nesting", "file": "m.py", "line": 8},  # same unit
        {"category": "Orphan Module", "file": "m.py", "line": 7},
        {"category": "Long File", "file": "m.py", "line": 0},
    ]
    targets = select_targets(findings, tmp_path)
    assert [(t.unit.name, t.category) for t in targets] == [("size", "Deep Nesting")]


def test_parse_reply():
    assert parse_reply("```python\nx = 1\n```") == ("x = 1\n", None)
    assert parse_reply("NO_CHANGE: false positive") == (None, "false positive")
    assert parse_reply("") == (None, None)


def test_splice_reindents_method_and_hoists_imports():
    unit = enclosing_unit(SOURCE, 7, "m.py")
    new_code = "import math\ndef size(self, a, b=1):\n    return math.floor(a + b) if a and b else 0\n"
    new, lo, hi = splice(SOURCE, unit, new_code)
    compile(new, "m.py", "exec")
    lines = new.splitlines()
    assert lines[2] == "import math"
    assert lines[lo - 1] == "    def size(self, a, b=1):"
    assert hi == lo + 1


def test_relocate_after_shift():
    unit = enclosing_unit(SOURCE, 14, "m.py")
    shifted = "import sys\nimport re\n" + SOURCE
    moved = relocate(shifted, unit)
    assert moved.start == unit.start + 2 and moved.name == "top"


def test_interface_preserved():
    unit = enclosing_unit(NESTED, 2)
    assert interface_preserved(NESTED, FLAT, unit)[0]
    ok, why = interface_preserved(NESTED, "def nested(a, *args):\n    return a\n", unit)
    assert not ok and "signature" in why
    ok, _ = interface_preserved(NESTED, "def other(a, b):\n    return a\n", unit)
    assert not ok


def test_gaming_flags():
    src = "import torch\n" + NESTED
    alias = "def nested(a, b):\n    T = torch.tensor\n    return T(a)\n"
    assert gaming_flags(NESTED, alias, src) == ["import_alias"]
    packed = "def nested(a, b):\n    x = a; y = b\n    return x + y  # noqa\n"
    assert gaming_flags(NESTED, packed, src) == [
        "statement_packing",
        "suppression_comment",
    ]
    assert gaming_flags(NESTED, FLAT, src) == []


def test_repetition_ratio():
    assert repetition_ratio("same line\n" * 50) > 0.9
    assert repetition_ratio("\n".join(f"line {i}" for i in range(50))) == 0.0


def _project(tmp_path):
    (tmp_path / "pkg.py").write_text(NESTED)
    (tmp_path / "test_pkg.py").write_text(
        "from pkg import nested\n\n"
        "def test_nested():\n"
        "    assert nested(1, 2) == 3\n"
        "    assert nested(0, 2) == 0\n"
    )
    return CheckCommand(
        argv=[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    )


def test_harness_rejects_gaming_then_accepts(tmp_path):
    test = _project(tmp_path)
    gamed = "```python\ndef nested(a, b):\n    if a:\n        if b:  # noqa\n            return a + b\n    return 0\n```"
    client = FakeClient([gamed, f"```python\n{FLAT}```"])
    h = Harness(root=tmp_path, client=client, detect=nesting_detector, test=test)
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    outcome = h.refactor(target, apply=True)
    assert outcome.status == "accepted"
    assert len(outcome.attempts) == 2
    first = outcome.attempts[0].checks
    assert first["gaming"] == ["suppression_comment"]
    assert not first["detectors"]["target_removed"]
    assert "suppression comment" in client.calls[1][-1]["content"]
    assert outcome.attempts[1].checks["behaviour"]["same"]
    assert (tmp_path / "pkg.py").read_text() == FLAT
    assert "if a and b" in outcome.diff


def test_harness_rejects_behaviour_change_and_restores_file(tmp_path):
    test = _project(tmp_path)
    broken = "```python\ndef nested(a, b):\n    return a + b\n```"
    client = FakeClient([broken])
    h = Harness(
        root=tmp_path, client=client, detect=nesting_detector, test=test, max_attempts=1
    )
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    outcome = h.refactor(target, apply=True)
    assert outcome.status == "rejected"
    assert outcome.attempts[0].checks["detectors"]["target_removed"]
    assert not outcome.attempts[0].checks["behaviour"]["same"]
    assert (tmp_path / "pkg.py").read_text() == NESTED


def test_harness_decline(tmp_path):
    (tmp_path / "pkg.py").write_text(NESTED)
    client = FakeClient(["NO_CHANGE: nesting is clearer here"])
    h = Harness(root=tmp_path, client=client, detect=nesting_detector)
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    outcome = h.refactor(target)
    assert outcome.status == "declined"
    assert outcome.attempts[0].declined == "nesting is clearer here"


def test_cli_refactor(tmp_path, monkeypatch):
    from code_review_agent import cli, refactor

    (tmp_path / "pkg.py").write_text(NESTED)
    monkeypatch.setattr(refactor, "default_detect", lambda tools: nesting_detector)
    monkeypatch.setattr(
        "code_review_agent.llm.LLMClient.complete_text",
        lambda self, messages, tools=None: CompletionResult(
            content=f"```python\n{FLAT}```"
        ),
    )
    out = tmp_path / "out" / "r.json"
    result = CliRunner().invoke(
        cli.main, ["refactor", str(tmp_path), "--apply", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "1/1 accepted" in result.output
    assert (tmp_path / "pkg.py").read_text() == FLAT
    assert out.with_suffix(".diff").read_text().startswith("--- a/pkg.py")
