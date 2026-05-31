"""
Supplemental, in-repo detectors that compensate for bugs in the installed
``code_quality_analyzer`` package (a git dependency we must not edit).

Two catalog smells are *wired* upstream but effectively never fire:

  * **Switch Statements** — upstream ``count_conditions`` only inspects the
    immediate ``orelse`` and does not recurse the ``elif`` chain, so the branch
    count caps at 2 (< the threshold of 3) and never triggers.
  * **Deep Inheritance Tree (DIT)** — upstream builds the inheritance graph with
    bare base-class names for edges but fully-qualified ``module.Class`` names
    for nodes, so the chain is broken and depth never exceeds the threshold
    (classes get mislabelled "Isolated" instead).

These pure-``ast`` helpers re-detect both correctly and return canonical smell
dicts (same shape as ``code_quality_analyzer`` smells) so the agent can merge
them into the normal findings list. Everything here is defensive and never
raises out to the caller.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _safe_parse(path: str) -> ast.AST | None:
    try:
        src = Path(path).read_text(encoding="utf-8", errors="replace")
        return ast.parse(src)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Switch Statements (long if/elif chains)
# ---------------------------------------------------------------------------


def _elif_children(tree: ast.AST) -> set[int]:
    """Ids of If nodes that are the ``elif`` continuation of another If."""
    children: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.If)
        ):
            children.add(id(node.orelse[0]))
    return children


def _branch_count(node: ast.If) -> int:
    """Count branches in an if/elif chain headed by ``node`` (recurses elif)."""
    count = 1
    current = node
    while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
        count += 1
        current = current.orelse[0]
    return count


def supplemental_switch_smells(py_files: list, file_threshold: int) -> list[dict[str, Any]]:
    """Detect long conditional (``Switch Statements``) chains across files."""
    smells: list[dict[str, Any]] = []
    threshold = int(file_threshold) if file_threshold is not None else 3
    for py_file in py_files:
        path = str(py_file)
        tree = _safe_parse(path)
        if tree is None:
            continue
        elif_ids = _elif_children(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if id(node) in elif_ids:
                continue  # part of a parent chain; counted at the head
            if not node.orelse:
                continue
            branches = _branch_count(node)
            if branches > threshold:
                smells.append(
                    {
                        "name": "Switch Statements",
                        "description": (
                            f"Complex conditional with {branches} branches at line "
                            f"{node.lineno} in {path}"
                        ),
                        "file_path": path,
                        "module_class": None,
                        "line_number": node.lineno,
                        "severity": "medium",
                    }
                )
    return smells


# ---------------------------------------------------------------------------
# Deep Inheritance Tree (DIT)
# ---------------------------------------------------------------------------


def _base_simple_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def supplemental_dit_smells(py_files: list, dit_threshold: int) -> list[dict[str, Any]]:
    """Detect classes with a deep inheritance tree across the analysed files."""
    threshold = int(dit_threshold) if dit_threshold is not None else 3

    bases: dict[str, list[str]] = {}
    location: dict[str, tuple[str, int]] = {}

    for py_file in py_files:
        path = str(py_file)
        tree = _safe_parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                resolved = [n for n in (_base_simple_name(b) for b in node.bases) if n]
                # First definition wins (stable, deterministic).
                bases.setdefault(node.name, resolved)
                location.setdefault(node.name, (path, node.lineno))

    framework_bases = {"object", "Exception", "dict", "list", "set", "tuple", "str", "int", "float"}

    def ancestors(name: str, seen: frozenset[str]) -> set[str]:
        if name in seen:
            return set()
        result: set[str] = set()
        for base in bases.get(name, []):
            if base in framework_bases:
                continue
            if "Mixin" in base or "Interface" in base or "ABC" in base or "Abstract" in base:
                continue
            if base in bases:  # only count locally-defined ancestors
                result.add(base)
                result |= ancestors(base, seen | {name})
        return result

    smells: list[dict[str, Any]] = []
    for name in bases:
        depth = len(ancestors(name, frozenset()))
        dit = depth + 1  # +1 for the implicit ``object`` root (matches upstream intent)
        if dit > threshold:
            path, lineno = location.get(name, ("Unknown", 0))
            severity = "high" if dit > threshold * 1.5 else "medium"
            smells.append(
                {
                    "name": "Deep Inheritance Tree (DIT)",
                    "description": f"Class '{name}' has DIT of {dit} in {path}",
                    "file_path": path,
                    "module_class": name,
                    "line_number": lineno,
                    "severity": severity,
                }
            )
    return smells


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def merge_dedup(existing: list, additions: list[dict]) -> list:
    """Append ``additions`` to ``existing``, dropping exact duplicate smells."""
    if not isinstance(existing, list):
        return existing
    seen: set[tuple] = set()
    for s in existing:
        if isinstance(s, dict):
            seen.add(
                (s.get("name"), s.get("file_path"), s.get("module_class"), s.get("line_number"))
            )
    for s in additions:
        key = (s.get("name"), s.get("file_path"), s.get("module_class"), s.get("line_number"))
        if key not in seen:
            existing.append(s)
            seen.add(key)
    return existing
