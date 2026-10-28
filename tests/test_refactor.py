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


def test_select_targets_covers_every_finding(tmp_path):
    (tmp_path / "m.py").write_text(SOURCE)
    findings = [
        {"category": "Deep Nesting", "file": "m.py", "line": 7},
        {"category": "Deep Nesting", "file": "m.py", "line": 8},  # same unit
        {"category": "Orphan Module", "file": "m.py", "line": 7},
        {"category": "Long File", "file": "m.py", "line": 0},
    ]
    targets = select_targets(findings, tmp_path)
    assert [(t.unit.kind, t.category, t.edit) for t in targets] == [
        ("function", "Deep Nesting", False),
        ("module", "Orphan Module", True),
    ]
    assert len(targets[0].related) == 1
    assert [f["category"] for f in targets[1].related] == ["Long File"]


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
    local = "class Value:\n    pass\n\n" + NESTED
    alias_local = "def nested(a, b):\n    V = Value\n    return V(a)\n"
    assert gaming_flags(NESTED, alias_local, local) == ["import_alias"]


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


SCRIPT = "import os\n\nLR = 0.001 * 3\nPATH = os.path.join('data', 'x')\n\n\ndef train(n):\n    return n * LR\n"


def magic_detector(root):
    """Flags `* 3` literals at any level, and a file-level rule while `seed` is absent."""
    out = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        rel = path.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if "* 3" in line:
                out.append(
                    {"category": "Magic number detected", "file": rel, "line": i}
                )
        if "seed" not in text:
            out.append(
                {"category": "Randomness Control Checker", "file": rel, "line": 0}
            )
    return out


def test_target_unit_statement_and_module():
    from code_review_agent.refactor import target_unit

    stmt = target_unit(SCRIPT, 3, "s.py")
    assert (stmt.kind, stmt.start, stmt.end, stmt.name) == (
        "statement",
        3,
        3,
        "LR = 0.001 * 3",
    )
    mod = target_unit(SCRIPT, 0, "s.py")
    assert (mod.kind, mod.start, mod.end, mod.name) == ("module", 1, 8, "s")
    assert target_unit(SCRIPT, 0, "s.py", max_module_lines=5) is None
    assert target_unit(SCRIPT, 8, "s.py").kind == "function"


def test_statement_and_module_interfaces():
    from code_review_agent.refactor import target_unit

    stmt = target_unit(SCRIPT, 3, "s.py")
    assert interface_preserved(
        "LR = 0.001 * 3\n", "_SCALE = 3\nLR = 0.001 * _SCALE\n", stmt
    )[0]
    ok, why = interface_preserved("LR = 0.001 * 3\n", "RATE = 0.003\n", stmt)
    assert not ok and "LR" in why
    mod = target_unit(SCRIPT, 0, "s.py")
    renamed = SCRIPT.replace("def train(n)", "def train(m)").replace("n * LR", "m * LR")
    ok, why = interface_preserved(SCRIPT, renamed, mod)
    assert not ok and "train" in why
    assert interface_preserved(
        SCRIPT, SCRIPT.replace("import os", "import os\nimport re"), mod
    )[0]


def test_select_targets_includes_statements_and_modules(tmp_path):
    (tmp_path / "s.py").write_text(SCRIPT)
    targets = select_targets(magic_detector(tmp_path), tmp_path)
    assert [(t.unit.kind, t.category) for t in targets] == [
        ("statement", "Magic number detected"),
        ("module", "Randomness Control Checker"),
    ]  # one module target per file


def test_harness_module_level_targets(tmp_path):
    (tmp_path / "s.py").write_text(SCRIPT)
    stmt_fix = "```python\n_SCALE = 3\nLR = 0.001 * _SCALE\n```"
    after_stmt = SCRIPT.replace("LR = 0.001 * 3", "_SCALE = 3\nLR = 0.001 * _SCALE")
    seeded = after_stmt.replace(
        "import os\n", "import os\nimport random\n\nrandom.seed(0)\n"
    )
    client = FakeClient([stmt_fix, f"```python\n{seeded}```"])
    h = Harness(root=tmp_path, client=client, detect=magic_detector)
    h.prepare()
    stmt, mod = select_targets(h.baseline_findings, tmp_path)
    o1 = h.refactor(stmt, apply=True)
    assert o1.status == "accepted", o1.attempts[0].checks
    assert (tmp_path / "s.py").read_text() == after_stmt
    o2 = h.refactor(mod, apply=True)
    assert o2.status == "accepted", o2.attempts[0].checks
    det = o2.attempts[0].checks["detectors"]
    assert det["before"] == {"Randomness Control Checker": 1} and det["after"] == {}
    assert (tmp_path / "s.py").read_text() == seeded


def test_related_findings_grouped_into_prompt(tmp_path):
    from code_review_agent.refactor import build_messages

    (tmp_path / "m.py").write_text(SOURCE)
    findings = [
        {"category": "Deep Nesting", "file": "m.py", "line": 7, "message": "nested"},
        {
            "category": "Magic number detected",
            "file": "m.py",
            "line": 9,
            "message": "1",
        },
    ]
    [t] = select_targets(findings, tmp_path)
    assert [f["category"] for f in t.related] == ["Magic number detected"]
    user = build_messages(t, SOURCE)[1]["content"]
    assert "Other findings in this unit" in user and "Magic number detected" in user


def test_select_targets_never_targets_test_code(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text(SOURCE)
    findings = [{"category": "Deep Nesting", "file": "tests/test_m.py", "line": 7}]
    assert select_targets(findings, tmp_path) == []


def test_class_attributes_are_interface():
    old = "class C:\n    name = 'x'\n    def f(self):\n        return 1\n"
    unit = enclosing_unit(old, 1)
    ok, why = interface_preserved(
        old, "class C:\n    def f(self):\n        return 1\n", unit
    )
    assert not ok and "name" in why
    assert interface_preserved(old, old.replace("return 1", "return int(True)"), unit)[
        0
    ]


def test_duplicate_definitions_and_collision_rejected(tmp_path):
    from code_review_agent.refactor import duplicate_definitions

    src = "class A:\n    def f(self):\n        return 1\n\n    def _h(self):\n        return 2\n"
    assert duplicate_definitions(src) == set()
    assert duplicate_definitions(
        src + "\n    def _h(self, x):\n        return x\n"
    ) == {"<module>.A._h"}

    (tmp_path / "pkg.py").write_text(
        NESTED + "\n\ndef _both(a, b):\n    return a and b\n"
    )
    clash = "```python\ndef _both(a, b):\n    return a and b\n\n\ndef nested(a, b):\n    if _both(a, b):\n        return a + b\n    return 0\n```"
    h = Harness(
        root=tmp_path,
        client=FakeClient([clash]),
        detect=nesting_detector,
        max_attempts=1,
    )
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    outcome = h.refactor(target)
    assert outcome.status == "rejected"
    assert outcome.attempts[0].checks["name_collisions"] == ["<module>._both"]


def test_parse_and_apply_edits():
    from code_review_agent.refactor import apply_edits, parse_edits

    files = {"pkg/m.py": "a = 1\nb = 2\n"}
    reply = (
        "FILE: pkg/m.py\n<<<<<<< SEARCH\nb = 2\n=======\nfrom ._b import b\n>>>>>>> REPLACE\n"
        "NEW FILE: pkg/_b.py\n```python\nb = 2\n```\n"
    )
    edits, new = parse_edits(reply)
    out, err = apply_edits(files.get, edits, new, {"pkg/m.py"}, "pkg")
    assert err is None
    assert out == {"pkg/m.py": "a = 1\nfrom ._b import b\n", "pkg/_b.py": "b = 2\n"}
    _, err = apply_edits(
        files.get, [("pkg/m.py", "c = 3", "")], {}, {"pkg/m.py"}, "pkg"
    )
    assert "matches 0 times" in err
    _, err = apply_edits(files.get, [], {"other/x.py": "x = 1"}, {"pkg/m.py"}, "pkg")
    assert "private module" in err


def test_harness_edit_mode_splits_module(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    big = "def keep(x):\n    return x\n\n\ndef move(y):\n    return y * 2\n"
    (tmp_path / "pkg" / "m.py").write_text(big)

    def long_file(root):
        out = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if len(path.read_text().splitlines()) > 5:
                out.append({"category": "Long File", "file": rel, "line": 0})
        return out

    reply = (
        "FILE: pkg/m.py\n<<<<<<< SEARCH\ndef move(y):\n    return y * 2\n=======\n"
        "from pkg._moved import move  # re-export\n>>>>>>> REPLACE\n"
        "NEW FILE: pkg/_moved.py\n```python\ndef move(y):\n    return y * 2\n```\n"
    )
    h = Harness(
        root=tmp_path, client=FakeClient([reply]), detect=long_file, max_attempts=1
    )
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    assert target.edit and target.unit.kind == "module"
    outcome = h.refactor(target, apply=True)
    assert outcome.status == "accepted", outcome.attempts[0].checks
    assert sorted(outcome.files) == ["pkg/_moved.py", "pkg/m.py"]
    assert (tmp_path / "pkg" / "_moved.py").exists()
    assert "def keep" in (tmp_path / "pkg" / "m.py").read_text()


def test_harness_edit_mode_rejects_lost_public_name(tmp_path):
    (tmp_path / "m.py").write_text(
        "def keep(x):\n    return x\n\n\ndef gone(y):\n    return y\n"
    )
    reply = "FILE: m.py\n<<<<<<< SEARCH\ndef gone(y):\n    return y\n=======\n>>>>>>> REPLACE\n"

    def detect(root):
        if "gone" in (root / "m.py").read_text():
            return [{"category": "Long File", "file": "m.py", "line": 0}]
        return []

    h = Harness(
        root=tmp_path, client=FakeClient([reply]), detect=detect, max_attempts=1
    )
    h.prepare()
    [target] = select_targets(h.baseline_findings, tmp_path)
    outcome = h.refactor(target)
    assert outcome.status == "rejected"
    assert "gone" in outcome.attempts[0].checks["interface_error"]


def test_safe_relpath_blocks_escape(tmp_path):
    import os

    from code_review_agent.refactor import safe_relpath

    (tmp_path / "pkg").mkdir()
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    os.symlink(outside, tmp_path / "pkg" / "link")
    assert safe_relpath(tmp_path, "pkg/m.py")
    assert not safe_relpath(tmp_path, "../x.py")
    assert not safe_relpath(tmp_path, "/etc/passwd")
    assert not safe_relpath(tmp_path, "pkg/link/x.py")


def test_edit_mode_refuses_files_outside_project(tmp_path):
    (tmp_path / "m.py").write_text("def keep(x):\n    return x\n")
    finding = {
        "category": "Duplicate Code",
        "file": "m.py",
        "line": 0,
        "related_files": ["../evil.py"],
    }
    reply = "FILE: ../evil.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"

    def detect(root):
        return [finding]

    h = Harness(
        root=tmp_path, client=FakeClient([reply]), detect=detect, max_attempts=1
    )
    h.prepare()
    [target] = select_targets([finding], tmp_path)
    outcome = h.refactor(target)
    assert outcome.status == "rejected"
    assert "not shown" in outcome.attempts[0].checks["edit_error"]
    assert not (tmp_path.parent / "evil.py").exists()


def test_symlinked_target_is_refused_and_untouched(tmp_path):
    import os

    outside = tmp_path.parent / "outside_pkg"
    outside.mkdir(exist_ok=True)
    (outside / "m.py").write_text(NESTED)
    os.symlink(outside, tmp_path / "pkg")
    finding = {"category": "Deep Nesting", "file": "pkg/m.py", "line": 2}
    h = Harness(
        root=tmp_path,
        client=FakeClient([f"```python\n{FLAT}```"]),
        detect=lambda r: [finding],
        max_attempts=1,
    )
    [target] = select_targets([finding], tmp_path)
    assert h.refactor(target).status == "error"
    assert (outside / "m.py").read_text() == NESTED
