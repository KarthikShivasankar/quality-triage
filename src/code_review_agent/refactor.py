"""
Detector-verified refactoring harness.

For each detector finding the LLM rewrites the smallest unit that contains it:
the enclosing function or class, else the module-level statement, or the whole
module for a file-level finding (small files only). The rewrite is spliced into
the file and kept only if every check passes:

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

from code_review_agent.llm import CompletionResult, LLMClient, complete_guarded
from code_review_agent.tools import is_test_file

# Rules about a whole module or several modules: always handled in edit mode on
# the module (the model may split it into new private modules, or decline).
CROSS_MODULE_RULES = re.compile(
    r"orphan module|hub-like|cyclic|unstable dependenc|god object|long file|"
    r"high lines of code|number of classes|classes per module|scattered|"
    r"potential improper api|redundant abstraction|parallel inheritance|"
    r"high fan-|alternative classes|unit testing checker",
    re.IGNORECASE,
)
UNFIXABLE_CATEGORIES = CROSS_MODULE_RULES  # former name

REFACTOR_SYSTEM_PROMPT = """You are refactoring Python code. A deterministic detector reported the finding below inside this code unit. Refactor the unit to remove the underlying design problem while preserving external behaviour exactly.

Rules:
- Keep the unit's name, parameters, defaults, decorators, return values, side effects, raised exceptions and public names. Never add *args/**kwargs.
- You may add private helper functions (or private methods, for a method or class) next to the unit, and imports at the top of your block.
- Improve the design the rule names; do not just make the detector stop matching. Not allowed: binding imported names to local aliases, suppression comments (noqa, pylint, mlscent: ignore, type: ignore), semicolons or packing statements onto one line, building strings from character codes, statements that only touch a name, deleting behaviour.
- For a module-level statement, keep every name it binds. For a whole module, keep every public top-level function and class (with their public methods and signatures) and every module-level name.
- If the finding is a false positive, or cannot be fixed inside this unit without changing its interface or other files, reply exactly: NO_CHANGE: <one-sentence reason>
- Otherwise reply with exactly one fenced ```python block holding the complete replacement for the unit (plus any helpers and imports) and nothing else."""

EDIT_SYSTEM_PROMPT = """You are refactoring Python code. A deterministic detector reported the finding below. The code is too large to rewrite in one piece, or the fix spans files, so you reply with exact edits. Remove the underlying design problem while preserving external behaviour exactly.

Reply format (one block per edit; SEARCH must be copied exactly from the current file and match exactly once):
FILE: <path relative to the project root>
<<<<<<< SEARCH
<exact existing lines>
=======
<replacement lines>
>>>>>>> REPLACE

To move code out of a module, you may create new private modules in the same directory:
NEW FILE: <same directory>/_<name>.py
```python
<module content>
```

Rules:
- Every public function, class, method, signature, class attribute and module-level name must stay importable from where it is now (re-export moved code with an import).
- Never add *args/**kwargs to existing callables. Only edit the files shown to you.
- Improve the design the rule names; do not just make the detector stop matching. Not allowed: binding imported names to local aliases, suppression comments (noqa, pylint, mlscent: ignore, type: ignore), semicolons or packing statements onto one line, building strings from character codes, statements that only touch a name, deleting behaviour, reusing an existing function or class name for a new helper.
- If the finding is a false positive, or cannot be fixed without changing behaviour or the public interface, reply exactly: NO_CHANGE: <one-sentence reason>"""

_EDIT_RE = re.compile(
    r"FILE:\s*(\S+)\s*\n<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S
)
_NEW_FILE_RE = re.compile(
    r"NEW FILE:\s*(\S+)\s*\n```(?:python|py)?[^\n]*\n(.*?)```", re.S
)
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
    kind: str  # "function" | "class" | "statement" | "module"
    indent: int


@dataclass
class Target:
    finding: dict[
        str, Any
    ]  # id, tool, category, severity, file, line, message, how_to_fix
    unit: Unit
    # other findings inside the same unit; the model is asked to fix them too
    related: list[dict[str, Any]] = field(default_factory=list)
    # True: the model answers with SEARCH/REPLACE edits (large unit, module-wide or
    # cross-file rule) instead of the whole unit
    edit: bool = False

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
    files: dict[str, str] = field(default_factory=dict)  # every touched/new file

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


def module_statement(source: str, line: int, file: str = "") -> Unit | None:
    """Top-level statement containing ``line``; ``name`` is its first line (for relocation)."""
    for node in ast.parse(source).body:
        start = min(
            [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))]
        )
        if start <= line <= node.end_lineno:
            first = source.splitlines()[start - 1].strip()[:80]
            return Unit(file, start, node.end_lineno, first, "statement", 0)
    return None


def target_unit(
    source: str,
    line: int,
    file: str = "",
    *,
    max_unit_lines: int = 200,
    max_module_lines: int = 250,
) -> Unit | None:
    """What to rewrite for a finding at ``line`` (0 means a file-level finding)."""
    if line <= 0:
        n = len(source.splitlines())
        if not 0 < n <= max_module_lines:
            return None
        return Unit(file, 1, n, Path(file).stem or "module", "module", 0)
    unit = enclosing_unit(source, line, file) or module_statement(source, line, file)
    if unit is None or unit.end - unit.start + 1 > max_unit_lines:
        return None
    return unit


def select_targets(
    findings: list[dict[str, Any]],
    root: str | Path,
    *,
    max_unit_lines: int = 200,
    max_module_lines: int = 250,
    include: re.Pattern[str] | None = None,
    exclude: re.Pattern[str] | None = None,
) -> list[Target]:
    """One target per unit: the first finding inside it, with the unit's other
    findings in ``related``. Every finding outside test code gets a target:

    - inside a function/class (else a module-level statement): that unit,
      rewritten whole up to ``max_unit_lines``, larger ones in edit mode;
    - file-level (line 0): the whole module, rewritten whole up to
      ``max_module_lines``, larger ones in edit mode;
    - module-wide or cross-module rules, and findings that name several files:
      the module in edit mode (the other files are shown and may be edited).
    """
    root = Path(root)
    sources: dict[str, str] = {}
    seen: dict[tuple[str, int, str], Target] = {}
    out: list[Target] = []
    for f in findings:
        cat = f.get("category", "")
        if f.get("line", 0) < 0 or not f.get("file") or is_test_file(f["file"]):
            continue  # test code is never refactored
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
        src = sources[f["file"]]
        if not src:
            continue
        n = len(src.splitlines())
        if (
            CROSS_MODULE_RULES.search(cat)
            or f.get("related_files")
            or f.get("line", 0) <= 0
        ):
            unit = Unit(f["file"], 1, n, Path(f["file"]).stem or "module", "module", 0)
            edit = n > max_module_lines or bool(
                CROSS_MODULE_RULES.search(cat) or f.get("related_files")
            )
            key = (f["file"], 0, "module")  # one module target per file, all its rules
        else:
            unit = enclosing_unit(src, f["line"], f["file"]) or module_statement(
                src, f["line"], f["file"]
            )
            if unit is None:
                continue
            edit = unit.end - unit.start + 1 > max_unit_lines
            key = (f["file"], unit.start, unit.kind)
        if key in seen:
            seen[key].related.append(dict(f))
            continue
        seen[key] = Target(finding=dict(f), unit=unit, edit=edit)
        out.append(seen[key])
    return out


def relocate(source: str, unit: Unit) -> Unit | None:
    """Find ``unit`` again after earlier edits shifted lines (same name and kind, nearest start)."""
    if unit.kind == "module":
        return Unit(unit.file, 1, len(source.splitlines()), unit.name, "module", 0)
    if unit.kind == "statement":
        found = None
        for node in ast.parse(source).body:
            start = min(
                [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))]
            )
            first = source.splitlines()[start - 1].strip()[:80]
            if first == unit.name and (
                found is None or abs(start - unit.start) < abs(found.start - unit.start)
            ):
                found = Unit(unit.file, start, node.end_lineno, first, "statement", 0)
        return found
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


def safe_relpath(root: Path, rel: str) -> bool:
    """``rel`` names a location strictly inside ``root``: relative, no ``..``, and no
    symlink on the way (model output chooses edit paths, so this guards every write)."""
    p = Path(rel)
    if not rel or p.is_absolute() or ".." in p.parts:
        return False
    root = Path(root).resolve()
    cur = root
    for part in p.parts:
        cur = cur / part
        if cur.is_symlink():
            return False
    try:
        (root / p).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def parse_edits(text: str) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """SEARCH/REPLACE edits and NEW FILE blocks from an edit-mode reply."""
    edits = [(p.strip(), s_, r) for p, s_, r in _EDIT_RE.findall(text or "")]
    new_files = {p.strip(): c for p, c in _NEW_FILE_RE.findall(text or "")}
    return edits, new_files


def apply_edits(
    read: Callable[[str], str | None],
    edits: list[tuple[str, str, str]],
    new_files: dict[str, str],
    allowed: set[str],
    new_dir: str,
) -> tuple[dict[str, str], str | None]:
    """New contents of every touched file, or an error explaining why not.

    ``read(path)`` returns the current text of a project file (None if absent).
    Edits may touch only ``allowed`` files; new files must be new private
    modules (``_name.py``) in ``new_dir``.
    """
    out: dict[str, str] = {}
    for path, search, replace in edits:
        if path not in allowed:
            return {}, f"edit to a file that was not shown: {path}"
        text = out[path] if path in out else read(path)
        hits = (text or "").count(search) if search else 0
        if hits != 1:
            return {}, (
                f"SEARCH block for {path} matches {hits} times, it must match "
                f"exactly once: {search[:120]!r}"
            )
        out[path] = text.replace(search, replace, 1)
    for path, code in new_files.items():
        p = Path(path)
        if (
            p.parent.as_posix() != new_dir
            or not p.name.startswith("_")
            or p.suffix != ".py"
        ):
            return (
                {},
                f"new file must be a private module (_name.py) in {new_dir}/: {path}",
            )
        if read(path) is not None:
            return {}, f"new file already exists: {path}"
        out[path] = code if code.endswith("\n") else code + "\n"
    if not out:
        return {}, "no FILE/SEARCH/REPLACE or NEW FILE blocks found"
    return out, None


def splice(source: str, unit: Unit, new_code: str) -> tuple[str, int, int]:
    """Replace the unit with ``new_code``; hoist top-level imports to the module header.

    Returns (new_source, new_unit_start, new_unit_end) where the span covers the
    whole spliced block (unit plus helpers).
    """
    block = textwrap.dedent(new_code).strip("\n") + "\n"
    if unit.kind == "module":
        return block, 1, len(block.splitlines())
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
        attrs = sorted(
            n.id
            for stmt in node.body
            if isinstance(stmt, (ast.Assign, ast.AnnAssign))
            for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
            for n in ast.walk(t)
            if isinstance(n, ast.Name) and not n.id.startswith("_")
        )
        return ("class", node.name, [ast.dump(b) for b in node.bases], methods, attrs)
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


def _bound_names(tree: ast.Module) -> set[str]:
    """Names a module body binds at top level."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def _module_interface(
    old: ast.Module, new: ast.Module, whole: bool
) -> tuple[bool, str]:
    missing = sorted(_bound_names(old) - _bound_names(new))
    if whole:
        missing = [m for m in missing if not m.startswith("_")]
    if missing:
        return False, f"module-level names no longer bound: {missing}"
    if not whole:
        return True, ""
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    new_defs = {n.name: n for n in new.body if isinstance(n, defs)}
    changed = [
        n.name
        for n in old.body
        if isinstance(n, defs)
        and not n.name.startswith("_")
        and not _same_interface(n, new_defs.get(n.name))
    ]
    if changed:
        return False, f"public definitions changed or removed: {changed}"
    return True, ""


def _same_interface(old: ast.AST, new: ast.AST | None) -> bool:
    if new is None:
        return False
    o, n = _signature(old), _signature(new)
    if o[0] == "class" and n[0] == "class":
        return (
            o[2] == n[2]
            and all(n[3].get(m) == sig for m, sig in o[3].items())
            and set(o[4]) <= set(n[4])
        )
    return o == n


def interface_preserved(old_unit: str, new_block: str, unit: Unit) -> tuple[bool, str]:
    """The block must define the unit with the same interface; class units keep public methods."""
    try:
        old = ast.parse(textwrap.dedent(old_unit))
        new = ast.parse(textwrap.dedent(new_block))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    if unit.kind in ("statement", "module"):
        return _module_interface(old, new, unit.kind == "module")
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
        lost = sorted(set(o[4]) - set(n[4]))
        if lost:
            return False, f"public class attributes removed: {lost}"
        return True, ""
    if o != n:
        return False, "signature or decorators changed"
    return True, ""


def duplicate_definitions(source: str) -> set[str]:
    """``scope.name`` of functions/classes defined more than once in the same scope
    (module or class body); a later definition silently replaces the earlier one."""
    dups: set[str] = set()

    def visit(body: list[ast.stmt], scope: str) -> None:
        seen: set[str] = set()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen and not node.decorator_list:
                    dups.add(f"{scope}.{node.name}")
                seen.add(node.name)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{scope}.{node.name}")

    try:
        visit(ast.parse(source).body, "<module>")
    except SyntaxError:
        pass
    return dups


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
    touched: set[str] | None = None,
) -> dict[str, Any]:
    """Compare detector findings in the unit before and after the rewrite.

    ``touched`` (edit mode): new rules are counted over every changed or new file,
    since moved code may carry its findings to another file."""
    u = target.unit

    def count(fs: list[dict], lo: int, hi: int) -> Counter:
        return Counter(
            f["category"] for f in fs if f["file"] == u.file and lo <= f["line"] <= hi
        )

    file_b = Counter(f["category"] for f in before if f["file"] == u.file)
    file_a = Counter(f["category"] for f in after if f["file"] == u.file)
    if u.kind == "module":  # file-level findings carry line 0
        b, a = file_b, file_a
    else:
        b = count(before, u.start, u.end)
        a = count(after, *new_span)
    if touched:
        tb = Counter(f["category"] for f in before if f["file"] in touched)
        ta = Counter(f["category"] for f in after if f["file"] in touched)
        new_rules = sorted(k for k in ta if ta[k] > tb.get(k, 0))
    else:
        new_rules = sorted(k for k in a if a[k] > b.get(k, 0))
    return {
        "target_removed": a[target.category] < b[target.category],
        "before": dict(b),
        "after": dict(a),
        "new_rules": new_rules,
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
    label = {
        "module": f"Whole module ({u.file})",
        "statement": f"Module-level statement ({u.file}, lines {u.start}-{u.end})",
    }.get(u.kind, f"Code unit ({u.file}, lines {u.start}-{u.end})")
    header = f"{label}:\n```python\n{code}```"
    imports = "".join(
        ln
        for ln in source.splitlines(keepends=True)[: _import_insertion_line(source)]
        if ln.startswith(("import ", "from "))
    )
    context = (
        f"Module imports:\n```python\n{imports}```\n\n"
        if imports and u.kind != "module"
        else ""
    )
    if grounded:
        keys = ("tool", "category", "severity", "file", "line", "message", "how_to_fix")
        finding = {k: target.finding.get(k) for k in keys}
        user = (
            f"Finding:\n```json\n{json.dumps(finding, indent=1, default=str)}\n```\n\n"
        )
        if target.related:
            others = [
                {k: f.get(k) for k in ("category", "line", "message")}
                for f in target.related
            ]
            user += (
                "Other findings in this unit; fix them too where you can do so "
                f"honestly:\n```json\n{json.dumps(others, indent=1, default=str)}\n```\n\n"
            )
        user += f"{context}{header}"
    else:
        user = f"{context}{header}"
    return [
        {"role": "system", "content": REFACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_edit_messages(
    target: Target, read: Callable[[str], str | None], max_related_lines: int = 800
) -> list[dict[str, str]]:
    """Edit-mode prompt: the finding(s), the target code and any related files."""
    u = target.unit
    source = read(u.file) or ""
    keys = ("tool", "category", "severity", "file", "line", "message", "how_to_fix")
    finding = {k: target.finding.get(k) for k in keys}
    user = f"Finding:\n```json\n{json.dumps(finding, indent=1, default=str)}\n```\n\n"
    if target.related:
        others = [
            {k: f.get(k) for k in ("category", "line", "message")}
            for f in target.related
        ]
        user += (
            "Other findings here; fix them too where you can do so honestly:\n"
            f"```json\n{json.dumps(others, indent=1, default=str)}\n```\n\n"
        )
    if u.kind == "module":
        user += f"File `{u.file}`:\n```python\n{source}```\n\n"
    else:
        user += (
            f"Code unit in `{u.file}` (lines {u.start}-{u.end}); SEARCH text must be "
            f"copied from it:\n```python\n{unit_source(source, u)}```\n\n"
        )
    for rel in edit_files(target)[1:]:
        text = read(rel)
        if text is None:
            continue
        lines = text.splitlines(keepends=True)
        cut = (
            ""
            if len(lines) <= max_related_lines
            else f" (first {max_related_lines} lines)"
        )
        user += f"Related file `{rel}`{cut}:\n```python\n{''.join(lines[:max_related_lines])}```\n\n"
    user += (
        f"New private modules may be created in `{Path(u.file).parent.as_posix()}/`."
    )
    return [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def edit_files(target: Target) -> list[str]:
    """Files an edit-mode reply may change: the target file, then related files."""
    files = [target.unit.file]
    for f in [target.finding, *target.related]:
        for rel in f.get("related_files") or []:
            if rel not in files and not is_test_file(rel):
                files.append(rel)
    return files


def _node_at(source: str, unit: Unit) -> ast.AST | None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            if start == unit.start and node.name == unit.name:
                return node
    return None


def feedback_message(attempt: Attempt) -> str:
    c = attempt.checks
    problems = []
    edit = c.get("mode") == "edit"
    if attempt.code is None and attempt.declined is None:
        problems.append(
            "No FILE/SEARCH/REPLACE blocks or NO_CHANGE line were found in your reply."
            if edit
            else "No ```python block or NO_CHANGE line was found in your reply."
        )
    if c.get("edit_error"):
        problems.append(f"The edits could not be applied: {c['edit_error']}")
    if c.get("compiles") is False:
        problems.append(f"The patched file does not compile: {c.get('compile_error')}")
    if c.get("interface") is False:
        problems.append(f"The interface changed: {c.get('interface_error')}")
    for name in c.get("name_collisions", []):
        problems.append(
            f"`{name}` is now defined twice in the same scope; the later definition "
            "would replace the earlier one. Give new helpers unique names."
        )
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
    again = (
        "corrected FILE/SEARCH/REPLACE blocks (SEARCH copied from the ORIGINAL code shown first)"
        if edit
        else "the complete replacement unit in one ```python block"
    )
    return (
        "Your rewrite was rejected:\n- "
        + "\n- ".join(problems)
        + f"\n\nFix these problems and reply with {again}, or NO_CHANGE: <reason> if the "
        "finding cannot be fixed honestly."
    )


def edit_interface(
    target: Target, old: dict[str, str | None], new: dict[str, str]
) -> tuple[bool, str]:
    """Every public name of every changed file stays importable with the same
    interface; definitions may move into a new file if the old module imports them."""
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    moved: dict[str, ast.AST] = {}
    for f, text in new.items():
        if old.get(f) is None:
            for node in ast.parse(text).body:
                if isinstance(node, defs):
                    moved.setdefault(node.name, node)
    u = target.unit
    for f, text in new.items():
        if old.get(f) is None:
            continue
        o, n = ast.parse(old[f]), ast.parse(text)
        lost = sorted(
            m for m in _bound_names(o) - _bound_names(n) if not m.startswith("_")
        )
        if lost:
            return False, f"{f}: public names no longer bound: {lost}"
        here = {d.name: d for d in n.body if isinstance(d, defs)}
        for d in o.body:
            if isinstance(d, defs) and not d.name.startswith("_"):
                if not _same_interface(d, here.get(d.name) or moved.get(d.name)):
                    if d.name in here or d.name in moved:
                        return False, f"{f}: interface of `{d.name}` changed"
        if f == u.file and u.kind in ("function", "class"):
            nu = relocate(text, u)
            if nu is None or not _same_interface(
                _node_at(old[f], u), _node_at(text, nu)
            ):
                return False, f"interface of `{u.name}` changed or it was removed"
    return True, ""


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
                "related_files": f.extra.get("related_files"),
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
        """One call; an empty or cut-off answer is retried (silent-failure guard)."""
        return complete_guarded(self.client, messages)

    def _read(self, rel: str) -> str | None:
        if not safe_relpath(self.root, rel):
            return None
        path = self.root / rel
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def check(
        self, target: Target, source: str, attempt: Attempt
    ) -> dict[str, str] | None:
        """Fill ``attempt.checks``; return {file: new text} if every check passes."""
        c = attempt.checks
        c["category"] = target.category
        if attempt.code is None:
            return None
        u = target.unit
        if target.edit:
            c["mode"] = "edit"
            edits, new_files = parse_edits(attempt.code)
            allowed = {f for f in edit_files(target) if safe_relpath(self.root, f)}
            files, err = apply_edits(
                self._read, edits, new_files, allowed, Path(u.file).parent.as_posix()
            )
            if err:
                c["edit_error"] = err
                return None
        else:
            new_source, lo, hi = splice(source, u, attempt.code)
            files = {u.file: new_source}
        # every write target must lie inside the project, never through a symlink;
        # checked before anything is written, so restore never touches other paths
        unsafe = [f for f in files if not safe_relpath(self.root, f)]
        if unsafe:
            c["edit_error"] = (
                f"paths outside the project or through a symlink: {unsafe}"
            )
            return None
        old = {f: self._read(f) for f in files}
        for f, text in files.items():
            try:
                compile(text, f, "exec")
            except SyntaxError as exc:
                c["compiles"], c["compile_error"] = False, f"{f}: {exc}"
                return None
        c["compiles"] = True
        if target.edit:
            ok, why = edit_interface(target, old, files)
            flags: set[str] = set()
            for f, text in files.items():
                flags |= set(gaming_flags(old[f] or "", text, old[f] or text))
            c["gaming"] = sorted(flags)
            if u.kind == "module":
                lo, hi = (
                    1,
                    len(files[u.file].splitlines()) if u.file in files else u.end,
                )
            else:
                nu = relocate(files.get(u.file, source), u)
                lo, hi = (nu.start, nu.end) if nu else (u.start, u.end)
        else:
            old_unit = unit_source(source, u)
            ok, why = interface_preserved(old_unit, attempt.code, u)
            c["gaming"] = gaming_flags(old_unit, attempt.code, source)
        c["interface"] = ok
        if not ok:
            c["interface_error"] = why
        c["name_collisions"] = sorted(
            set().union(
                *(
                    duplicate_definitions(t) - duplicate_definitions(old[f] or "")
                    for f, t in files.items()
                )
            )
        )
        c["span"] = [lo, hi]
        if target.edit:
            c["files"] = sorted(files)
        try:
            for f, text in files.items():
                (self.root / f).write_text(text, encoding="utf-8")
            after = self.detect(self.root)
            c["detectors"] = rule_delta(
                self.baseline_findings or [],
                after,
                target,
                (lo, hi),
                touched=set(files) if target.edit else None,
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
            for f in files:
                if old[f] is None:
                    (self.root / f).unlink(missing_ok=True)
                else:
                    (self.root / f).write_text(old[f], encoding="utf-8")
        det = c["detectors"]
        attempt.passed = (
            c["interface"]
            and not c["gaming"]
            and not c["name_collisions"]
            and not det["new_rules"]
            and (det["target_removed"] or not self.require_removal)
            and (c.get("behaviour") or {"same": True})["same"]
        )
        return files if attempt.passed else None

    def refactor(self, target: Target, apply: bool = False) -> Outcome:
        self.prepare()
        if not safe_relpath(self.root, target.unit.file):
            return Outcome(
                target, "error", [Attempt("", None, None, {"error": "unsafe path"})]
            )
        path = self.root / target.unit.file
        source = path.read_text(encoding="utf-8")
        unit = relocate(source, target.unit)
        if unit is None:
            return Outcome(
                target, "error", [Attempt("", None, None, {"error": "unit not found"})]
            )
        target.unit = unit
        messages = (
            build_edit_messages(target, self._read)
            if target.edit
            else build_messages(target, source, self.grounded)
        )
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
            if target.edit:
                edits, new_files = parse_edits(res.content)
                code = res.content if (edits or new_files) else None
                declined = None if code else parse_reply(res.content)[1]
            else:
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
            files = self.check(target, source, att)
            if files is not None:
                old = {f: self._read(f) or "" for f in files}
                if apply:
                    for f, text in files.items():
                        (self.root / f).write_text(text, encoding="utf-8")
                    self.baseline_findings = self.detect(self.root)
                diff = "".join(
                    "".join(
                        difflib.unified_diff(
                            old[f].splitlines(keepends=True),
                            text.splitlines(keepends=True),
                            f"a/{f}",
                            f"b/{f}",
                        )
                    )
                    for f, text in files.items()
                )
                return Outcome(
                    target,
                    "accepted",
                    attempts,
                    files.get(target.unit.file),
                    diff,
                    files,
                )
            messages = messages + [
                {"role": "assistant", "content": res.content or "(empty reply)"},
                {"role": "user", "content": feedback_message(att)},
            ]
        return Outcome(target, "rejected", attempts)
