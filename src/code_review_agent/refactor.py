"""
Detector-verified refactoring harness.

For each detector finding inside a function or class, the LLM rewrites that
unit and the rewrite is spliced into the file. It is kept only if every check
passes:

1. the file parses and the unit's interface (name, parameters, decorators,
   public methods) is unchanged;
2. the targeted rule fires less often in the unit and no other rule fires more;
3. no known detector-gaming pattern was introduced (aliasing imported names,
   suppression comments, statement packing, character-code strings, no-op
   statements);
4. the project's behaviour check (usually its test suite) gives the same result
   as on the original code.

Failed checks go back to the model for another attempt. The model may decline a
finding it judges a false positive or not fixable inside the unit; a decline is
a valid outcome, not a failure.
"""

from __future__ import annotations

import ast
import difflib
import io
import json
import re
import shlex
import subprocess
import textwrap
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from code_review_agent.llm import CompletionResult, LLMClient

# Whole-module or cross-module findings cannot be fixed by rewriting one unit.
UNFIXABLE_CATEGORIES = re.compile(
    r"orphan module|hub-like|cyclic|unstable dependenc|god object|long file|"
    r"high lines of code|number of classes|classes per module|scattered|"
    r"potential improper api|redundant abstraction",
    re.IGNORECASE,
)

REFACTOR_SYSTEM_PROMPT = """You are refactoring Python code. A deterministic detector reported the finding below inside this code unit. Refactor the unit to remove the underlying design problem while preserving external behaviour exactly.

Rules:
- Keep the unit's name, parameters, defaults, decorators, return values, side effects, raised exceptions and public names. Never add *args/**kwargs.
- You may add private helper functions (or private methods, for a method or class) next to the unit, and imports at the top of your block.
- Improve the design the rule names; do not just make the detector stop matching. Not allowed: binding imported names to local aliases, suppression comments (noqa, pylint, mlscent: ignore, type: ignore), semicolons or packing statements onto one line, building strings from character codes, statements that only touch a name, deleting behaviour.
- If the finding is a false positive, or cannot be fixed inside this unit without changing its interface or other files, reply exactly: NO_CHANGE: <one-sentence reason>
- Otherwise reply with exactly one fenced ```python block holding the complete replacement for the unit (plus any helpers and imports) and nothing else."""

_PY_BLOCK = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.S)
_DECLINE = re.compile(r"^\s*NO_CHANGE\s*:\s*(.+)", re.I | re.M)
_SUPPRESS = re.compile(
    r"#\s*(noqa|pylint\s*:\s*disable|mlscent\s*:\s*ignore|type\s*:\s*ignore|nosec|pragma)",
    re.I,
)
_PYTEST_COUNTS = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    file: str  # relative to the project root
    start: int  # first line, including decorators
    end: int
    name: str
    kind: str  # "function" | "class"
    indent: int


@dataclass
class Target:
    finding: dict[
        str, Any
    ]  # id, tool, category, severity, file, line, message, how_to_fix
    unit: Unit

    @property
    def category(self) -> str:
        return self.finding["category"]


@dataclass
class Attempt:
    reply: str
    code: str | None
    declined: str | None
    checks: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning_chars: int = 0
    repetition: float = 0.0


@dataclass
class Outcome:
    target: Target
    status: str  # accepted | declined | rejected | error
    attempts: list[Attempt]
    new_source: str | None = None
    diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Units and targets
# ---------------------------------------------------------------------------


def enclosing_unit(source: str, line: int, file: str = "") -> Unit | None:
    """Smallest function (preferred) or class whose span contains ``line``."""
    best = None
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        if not start <= line <= node.end_lineno:
            continue
        is_class = isinstance(node, ast.ClassDef)
        key = (is_class, node.end_lineno - start)
        if best is None or key < best[0]:
            best = (key, node, start)
    if best is None:
        return None
    _, node, start = best
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    return Unit(file, start, node.end_lineno, node.name, kind, node.col_offset)


def select_targets(
    findings: list[dict[str, Any]],
    root: str | Path,
    *,
    max_unit_lines: int = 120,
    include: re.Pattern[str] | None = None,
    exclude: re.Pattern[str] | None = UNFIXABLE_CATEGORIES,
) -> list[Target]:
    """One target per (file, unit): the first line-level finding inside it."""
    root = Path(root)
    sources: dict[str, str] = {}
    seen: set[tuple[str, int]] = set()
    out: list[Target] = []
    for f in findings:
        cat = f.get("category", "")
        if f.get("line", 0) <= 0 or not f.get("file"):
            continue
        if include and not include.search(cat):
            continue
        if exclude and exclude.search(cat):
            continue
        if f["file"] not in sources:
            try:
                sources[f["file"]] = (root / f["file"]).read_text(encoding="utf-8")
                ast.parse(sources[f["file"]])
            except (OSError, SyntaxError, UnicodeDecodeError):
                sources[f["file"]] = ""
        if not sources[f["file"]]:
            continue
        unit = enclosing_unit(sources[f["file"]], f["line"], f["file"])
        if unit is None or unit.end - unit.start + 1 > max_unit_lines:
            continue
        if (f["file"], unit.start) in seen:
            continue
        seen.add((f["file"], unit.start))
        out.append(Target(finding=dict(f), unit=unit))
    return out


def relocate(source: str, unit: Unit) -> Unit | None:
    """Find ``unit`` again after earlier edits shifted lines (same name and kind, nearest start)."""
    best = None
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if (
            node.name != unit.name
            or kind != unit.kind
            or node.col_offset != unit.indent
        ):
            continue
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        if best is None or abs(start - unit.start) < abs(best.start - unit.start):
            best = Unit(unit.file, start, node.end_lineno, unit.name, kind, unit.indent)
    return best


def unit_source(source: str, unit: Unit) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[unit.start - 1 : unit.end])


# ---------------------------------------------------------------------------
# Model reply -> patched file
# ---------------------------------------------------------------------------


def parse_reply(text: str) -> tuple[str | None, str | None]:
    """(code, decline_reason) from a model reply; at most one is set."""
    blocks = _PY_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len), None
    m = _DECLINE.search(text or "")
    if m:
        return None, m.group(1).strip()
    return None, None


def splice(source: str, unit: Unit, new_code: str) -> tuple[str, int, int]:
    """Replace the unit with ``new_code``; hoist top-level imports to the module header.

    Returns (new_source, new_unit_start, new_unit_end) where the span covers the
    whole spliced block (unit plus helpers).
    """
    block = textwrap.dedent(new_code).strip("\n") + "\n"
    imports: list[str] = []
    try:
        tree = ast.parse(block)
    except SyntaxError:
        tree = None
    if tree is not None:
        blines = block.splitlines(keepends=True)
        keep = [True] * len(blines)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append("".join(blines[node.lineno - 1 : node.end_lineno]))
                for i in range(node.lineno - 1, node.end_lineno):
                    keep[i] = False
        block = "".join(ln for ln, k in zip(blines, keep) if k).strip("\n") + "\n"
    body = textwrap.indent(block, " " * unit.indent)
    lines = source.splitlines(keepends=True)
    head, tail = lines[: unit.start - 1], lines[unit.end :]
    have = {ln.strip() for ln in head}
    imports = [i for i in imports if i.strip() not in have]
    insert_at = _import_insertion_line(source) if imports else 0
    if imports and insert_at <= len(head):
        head = head[:insert_at] + imports + head[insert_at:]
    start = len(head) + 1
    new = "".join(head) + body + "".join(tail)
    return new, start, start + len(body.splitlines()) - 1


def _import_insertion_line(source: str) -> int:
    """0-based index after the last top-level import (or after the docstring)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            idx = node.end_lineno
        elif (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
        ):
            idx = max(idx, node.end_lineno)
    return idx


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _signature(node: ast.AST) -> Any:
    if isinstance(node, ast.ClassDef):
        methods = {
            n.name: _signature(n)
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (not n.name.startswith("_") or n.name.startswith("__"))
        }
        return ("class", node.name, [ast.dump(b) for b in node.bases], methods)
    a = node.args
    return (
        type(node).__name__,
        node.name,
        [p.arg for p in a.posonlyargs],
        [p.arg for p in a.args],
        a.vararg.arg if a.vararg else None,
        [p.arg for p in a.kwonlyargs],
        a.kwarg.arg if a.kwarg else None,
        [ast.dump(d) for d in a.defaults],
        [ast.dump(d) if d else None for d in a.kw_defaults],
        [ast.dump(d) for d in node.decorator_list],
    )


def interface_preserved(old_unit: str, new_block: str, unit: Unit) -> tuple[bool, str]:
    """The block must define the unit with the same interface; class units keep public methods."""
    try:
        old = ast.parse(textwrap.dedent(old_unit))
        new = ast.parse(textwrap.dedent(new_block))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    old_node = next(n for n in old.body if getattr(n, "name", None) == unit.name)
    match = [n for n in new.body if getattr(n, "name", None) == unit.name]
    if len(match) != 1:
        return False, f"the block must define `{unit.name}` exactly once"
    o, n = _signature(old_node), _signature(match[0])
    if o[0] == "class" and n[0] == "class":
        if o[2] != n[2]:
            return False, "class bases changed"
        missing = [m for m in o[3] if m not in n[3]]
        changed = [m for m in o[3] if m in n[3] and o[3][m] != n[3][m]]
        if missing or changed:
            return False, f"public methods removed {missing} or changed {changed}"
        return True, ""
    if o != n:
        return False, "signature or decorators changed"
    return True, ""


def _statements_per_line(code: str) -> int:
    """Number of ';' separators (statement packing)."""
    try:
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        return sum(1 for t in toks if t.type == tokenize.OP and t.string == ";")
    except (tokenize.TokenError, IndentationError):
        return 0


def _imported_names(source: str) -> set[str]:
    """Imported names plus module-level classes and functions (alias targets)."""
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    names |= {
        n.name
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return names


def _gaming_counts(code: str, imported: set[str]) -> Counter:
    c: Counter = Counter()
    c["suppression_comment"] = len(_SUPPRESS.findall(code))
    c["statement_packing"] = _statements_per_line(code)
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return c
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "chr":
            c["char_code_string"] += 1
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt, val = node.targets[0], node.value
            if isinstance(tgt, ast.Name) and isinstance(val, (ast.Name, ast.Attribute)):
                root = val
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in imported:
                    c["import_alias"] += 1
            if (
                isinstance(tgt, ast.Name)
                and tgt.id == "_"
                and isinstance(val, (ast.Name, ast.Attribute))
            ):
                c["no_op_statement"] += 1
        elif isinstance(node, ast.Expr) and isinstance(
            node.value, (ast.Name, ast.Attribute)
        ):
            c["no_op_statement"] += 1
    return c


def gaming_flags(old_unit: str, new_block: str, file_source: str) -> list[str]:
    """Detector-gaming patterns that appear more often after the rewrite."""
    imported = _imported_names(file_source) | _imported_names(
        textwrap.dedent(new_block)
    )
    before = _gaming_counts(old_unit, imported)
    after = _gaming_counts(new_block, imported)
    return sorted(k for k in after if after[k] > before[k])


def rule_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    target: Target,
    new_span: tuple[int, int],
) -> dict[str, Any]:
    """Compare detector findings in the unit before and after the rewrite."""
    u = target.unit

    def count(fs: list[dict], lo: int, hi: int) -> Counter:
        return Counter(
            f["category"] for f in fs if f["file"] == u.file and lo <= f["line"] <= hi
        )

    b = count(before, u.start, u.end)
    a = count(after, *new_span)
    file_b = Counter(f["category"] for f in before if f["file"] == u.file)
    file_a = Counter(f["category"] for f in after if f["file"] == u.file)
    return {
        "target_removed": a[target.category] < b[target.category],
        "before": dict(b),
        "after": dict(a),
        "new_rules": sorted(k for k in a if a[k] > b.get(k, 0)),
        "unit_findings_before": sum(b.values()),
        "unit_findings_after": sum(a.values()),
        "file_findings_before": sum(file_b.values()),
        "file_findings_after": sum(file_a.values()),
    }


# ---------------------------------------------------------------------------
# Behaviour check
# ---------------------------------------------------------------------------


def pytest_signature(rc: int, output: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for n, k in _PYTEST_COUNTS.findall(output):
        k = "errors" if k.startswith("error") else k
        counts[k] = int(n)
    return {"rc": rc, "counts": counts}


@dataclass
class CheckCommand:
    """A behaviour check: run ``argv`` in ``cwd`` (relative to the project root)."""

    argv: list[str]
    cwd: str = "."
    env: dict[str, str] | None = None
    timeout: int = 1800

    @classmethod
    def parse(cls, cmd: str, **kw: Any) -> "CheckCommand":
        return cls(argv=shlex.split(cmd), **kw)

    def run(self, root: str | Path) -> tuple[int, str]:
        import os

        env = {**os.environ, **(self.env or {})}
        try:
            p = subprocess.run(
                self.argv,
                cwd=Path(root) / self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as exc:
            return -9, f"TIMEOUT after {self.timeout}s\n{exc.stdout or ''}"
        except OSError as exc:
            return -1, str(exc)


def same_behaviour(base: dict[str, Any], now: dict[str, Any]) -> bool:
    """Same exit code and no fewer passes / no more failures than the baseline."""
    if base["rc"] != now["rc"]:
        return False
    b, n = base.get("counts", {}), now.get("counts", {})
    if not b:
        return True
    return (
        n.get("passed", 0) >= b.get("passed", 0)
        and n.get("failed", 0) <= b.get("failed", 0)
        and n.get("errors", 0) <= b.get("errors", 0)
    )


# ---------------------------------------------------------------------------
# Prompts and LLM hygiene
# ---------------------------------------------------------------------------


def repetition_ratio(text: str) -> float:
    """Share of non-blank lines that repeat an earlier line (reasoning loops)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 20:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def build_messages(
    target: Target, source: str, grounded: bool = True
) -> list[dict[str, str]]:
    u = target.unit
    code = unit_source(source, u)
    header = f"Code unit ({u.file}, lines {u.start}-{u.end}):\n```python\n{code}```"
    imports = "".join(
        ln
        for ln in source.splitlines(keepends=True)[: _import_insertion_line(source)]
        if ln.startswith(("import ", "from "))
    )
    context = f"Module imports:\n```python\n{imports}```\n\n" if imports else ""
    if grounded:
        keys = ("tool", "category", "severity", "file", "line", "message", "how_to_fix")
        finding = {k: target.finding.get(k) for k in keys}
        user = f"Finding:\n```json\n{json.dumps(finding, indent=1, default=str)}\n```\n\n{context}{header}"
    else:
        user = f"{context}{header}"
    return [
        {"role": "system", "content": REFACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def feedback_message(attempt: Attempt) -> str:
    c = attempt.checks
    problems = []
    if attempt.code is None and attempt.declined is None:
        problems.append("No ```python block or NO_CHANGE line was found in your reply.")
    if c.get("compiles") is False:
        problems.append(f"The patched file does not compile: {c.get('compile_error')}")
    if c.get("interface") is False:
        problems.append(f"The interface changed: {c.get('interface_error')}")
    for flag in c.get("gaming", []):
        problems.append(
            f"Detector-gaming pattern introduced: {flag.replace('_', ' ')}."
        )
    det = c.get("detectors") or {}
    if det and not det.get("target_removed"):
        problems.append(
            f"The detector still reports `{c.get('category')}` in the unit "
            f"(before {det['before']}, after {det['after']})."
        )
    if det.get("new_rules"):
        problems.append(f"New detector findings in the unit: {det['new_rules']}.")
    beh = c.get("behaviour") or {}
    if beh and not beh.get("same"):
        problems.append(
            "The behaviour check no longer matches the original "
            f"(before {beh.get('baseline')}, after {beh.get('result')}). Output tail:\n"
            f"```\n{beh.get('tail', '')}\n```"
        )
    return (
        "Your rewrite was rejected:\n- "
        + "\n- ".join(problems)
        + "\n\nFix these problems and reply with the complete replacement unit in one "
        "```python block, or NO_CHANGE: <reason> if the finding cannot be fixed honestly "
        "inside this unit."
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

Detect = Callable[[Path], list[dict[str, Any]]]


def default_detect(tools: tuple[str, ...] = ("python-smells", "ml-smells")) -> Detect:
    """Run the pipeline detectors and return plain finding dicts."""

    def detect(root: Path) -> list[dict[str, Any]]:
        from code_review_agent.pipeline import run_pipeline

        res = run_pipeline(str(root), tools=list(tools), parallel=False)
        return [
            {
                "id": f.finding_id,
                "tool": f.tool,
                "category": f.category,
                "severity": f.severity.value,
                "file": f.file,
                "line": f.line,
                "message": f.message,
                "how_to_fix": f.how_to_fix,
            }
            for f in res.report.findings
        ]

    return detect


@dataclass
class Harness:
    """Propose → check → feed back, for one project root, one target at a time."""

    root: Path
    client: LLMClient
    detect: Detect = field(default_factory=default_detect)
    test: CheckCommand | Callable[[Path], dict[str, Any]] | None = None
    max_attempts: int = 3
    grounded: bool = True
    require_removal: bool = True
    baseline_findings: list[dict[str, Any]] | None = None
    baseline_behaviour: dict[str, Any] | None = None
    on_step: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def note(self, msg: str) -> None:
        if self.on_step:
            self.on_step(msg)

    def behaviour(self) -> dict[str, Any] | None:
        if self.test is None:
            return None
        if callable(self.test) and not isinstance(self.test, CheckCommand):
            return self.test(self.root)
        rc, out = self.test.run(self.root)
        sig = pytest_signature(rc, out)
        sig["tail"] = out[-1500:]
        return sig

    def prepare(self) -> None:
        if self.baseline_findings is None:
            self.note("Running detectors on the original code…")
            self.baseline_findings = self.detect(self.root)
        if self.test is not None and self.baseline_behaviour is None:
            self.note("Running the behaviour check on the original code…")
            self.baseline_behaviour = self.behaviour()

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        """One call; an empty or truncated answer is retried once (silent-failure guard)."""
        res = self.client.complete_text(messages)
        if not res.content.strip() or res.finish_reason == "length":
            res = self.client.complete_text(messages)
        return res

    def check(self, target: Target, source: str, attempt: Attempt) -> str | None:
        """Fill ``attempt.checks``; return the patched source if all checks pass."""
        c = attempt.checks
        c["category"] = target.category
        if attempt.code is None:
            return None
        old_unit = unit_source(source, target.unit)
        new_source, lo, hi = splice(source, target.unit, attempt.code)
        try:
            compile(new_source, target.unit.file, "exec")
            c["compiles"] = True
        except SyntaxError as exc:
            c["compiles"], c["compile_error"] = False, str(exc)
            return None
        ok, why = interface_preserved(old_unit, attempt.code, target.unit)
        c["interface"] = ok
        if not ok:
            c["interface_error"] = why
        c["gaming"] = gaming_flags(old_unit, attempt.code, source)
        c["span"] = [lo, hi]
        path = self.root / target.unit.file
        path.write_text(new_source, encoding="utf-8")
        try:
            after = self.detect(self.root)
            c["detectors"] = rule_delta(
                self.baseline_findings or [], after, target, (lo, hi)
            )
            beh = self.behaviour()
            if beh is not None:
                same = (
                    beh.get("same")
                    if "same" in beh
                    else same_behaviour(self.baseline_behaviour or beh, beh)
                )
                c["behaviour"] = {
                    "same": bool(same),
                    "baseline": {
                        k: v
                        for k, v in (self.baseline_behaviour or {}).items()
                        if k != "tail"
                    },
                    "result": {k: v for k, v in beh.items() if k != "tail"},
                    "tail": beh.get("tail", "")[-800:],
                }
        finally:
            path.write_text(source, encoding="utf-8")
        det = c["detectors"]
        attempt.passed = (
            c["interface"]
            and not c["gaming"]
            and not det["new_rules"]
            and (det["target_removed"] or not self.require_removal)
            and (c.get("behaviour") or {"same": True})["same"]
        )
        return new_source if attempt.passed else None

    def refactor(self, target: Target, apply: bool = False) -> Outcome:
        self.prepare()
        path = self.root / target.unit.file
        source = path.read_text(encoding="utf-8")
        unit = relocate(source, target.unit)
        if unit is None:
            return Outcome(
                target, "error", [Attempt("", None, None, {"error": "unit not found"})]
            )
        target.unit = unit
        messages = build_messages(target, source, self.grounded)
        attempts: list[Attempt] = []
        for i in range(self.max_attempts):
            self.note(
                f"{target.unit.file}:{target.unit.start} `{target.category}` attempt {i + 1}"
            )
            try:
                res = self.complete(messages)
            except Exception as exc:  # transport failure: report, do not crash the run
                attempts.append(
                    Attempt(
                        reply="", code=None, declined=None, checks={"error": str(exc)}
                    )
                )
                return Outcome(target, "error", attempts)
            code, declined = parse_reply(res.content)
            att = Attempt(
                reply=res.content,
                code=code,
                declined=declined,
                finish_reason=res.finish_reason,
                usage=res.usage,
                reasoning_chars=len(res.reasoning),
                repetition=round(repetition_ratio(res.reasoning), 3),
            )
            attempts.append(att)
            if declined:
                return Outcome(target, "declined", attempts)
            new_source = self.check(target, source, att)
            if new_source is not None:
                if apply:
                    path.write_text(new_source, encoding="utf-8")
                    self.baseline_findings = self.detect(self.root)
                diff = "".join(
                    difflib.unified_diff(
                        source.splitlines(keepends=True),
                        new_source.splitlines(keepends=True),
                        f"a/{target.unit.file}",
                        f"b/{target.unit.file}",
                    )
                )
                return Outcome(target, "accepted", attempts, new_source, diff)
            messages = messages + [
                {"role": "assistant", "content": res.content or "(empty reply)"},
                {"role": "user", "content": feedback_message(att)},
            ]
        return Outcome(target, "rejected", attempts)
