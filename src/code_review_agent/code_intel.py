"""
Pure-Python AST-based code intelligence.
Provides symbol lookup, import graphs, find-usages, and per-function metrics.
All results include exact file:line:col locations.
"""

from __future__ import annotations

import ast
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Location:
    file: str           # absolute path
    line: int           # 1-based
    col: int            # 1-based
    end_line: int | None = None
    end_col: int | None = None

    def as_str(self, relative_to: str | None = None) -> str:
        f = os.path.relpath(self.file, relative_to) if relative_to else self.file
        return f"{f}:{self.line}:{self.col}"


@dataclass
class SymbolDef:
    name: str
    kind: str           # "function" | "async_function" | "class" | "method" | "async_method"
    location: Location
    docstring: str | None = None
    signature: str | None = None   # "(self, x: int) -> str"
    parent: str | None = None      # class name if this is a method
    is_private: bool = False


@dataclass
class ImportEdge:
    from_file: str      # absolute path
    module: str         # e.g. "os.path", "pandas"
    names: list[str]    # empty = "import module"; non-empty = "from X import Y, Z"
    alias: str | None   # "import X as Y"
    line: int


@dataclass
class Usage:
    symbol_name: str
    location: Location
    context_line: str   # the source line for quick inspection


@dataclass
class FunctionMetrics:
    name: str
    location: Location
    loc: int                     # non-blank, non-comment lines
    cyclomatic_complexity: int   # McCabe
    param_count: int
    nesting_depth: int
    return_count: int
    parent_class: str | None = None


@dataclass
class FileIntelligence:
    file: str           # absolute path
    symbols: list[SymbolDef] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    metrics: list[FunctionMetrics] = field(default_factory=list)
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(node: ast.AST) -> int:
    return getattr(node, "col_offset", 0) + 1   # 1-based


def _end(node: ast.AST) -> tuple[int | None, int | None]:
    return (
        getattr(node, "end_lineno", None),
        (getattr(node, "end_col_offset", None) or 0) + 1 if getattr(node, "end_col_offset", None) is not None else None,
    )


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a human-readable signature string."""
    args = node.args
    parts: list[str] = []

    # positional args
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, arg in enumerate(args.args):
        default_idx = i - (n_args - n_defaults)
        part = arg.arg
        if arg.annotation:
            part += f": {ast.unparse(arg.annotation)}"
        if default_idx >= 0:
            part += f" = {ast.unparse(args.defaults[default_idx])}"
        parts.append(part)

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kw in args.kwonlyargs:
        parts.append(kw.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    return f"({', '.join(parts)}){ret}"


def _cyclomatic(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """McCabe cyclomatic complexity for a function node."""
    count = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                               ast.With, ast.Assert, ast.comprehension)):
            count += 1
        elif isinstance(child, ast.BoolOp) and isinstance(child.op, (ast.And, ast.Or)):
            count += len(child.values) - 1
    return count


def _nesting_depth(node: ast.AST) -> int:
    """Max nesting depth within a function body."""
    depth = 0
    stack = [(node, 0)]
    while stack:
        n, d = stack.pop()
        if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try,
                          ast.ExceptHandler, ast.AsyncFor, ast.AsyncWith)):
            depth = max(depth, d + 1)
            d = d + 1
        for child in ast.iter_child_nodes(n):
            stack.append((child, d))
    return depth


def _loc(node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]) -> int:
    """Non-blank, non-comment lines within a function."""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    count = 0
    for line in source_lines[start:end]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _param_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    total = len(args.args) + len(args.kwonlyargs)
    if args.vararg:
        total += 1
    if args.kwarg:
        total += 1
    # Subtract 'self' / 'cls' for methods
    if args.args and args.args[0].arg in ("self", "cls"):
        total -= 1
    return total


# ---------------------------------------------------------------------------
# File analyser visitor
# ---------------------------------------------------------------------------

class _FileVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str, include_private: bool):
        self.file_path = file_path
        self.source_lines = source.splitlines()
        self.include_private = include_private
        self.symbols: list[SymbolDef] = []
        self.imports: list[ImportEdge] = []
        self.metrics: list[FunctionMetrics] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        if not self.include_private and node.name.startswith("_"):
            return
        end_l, end_c = _end(node)
        self.symbols.append(SymbolDef(
            name=node.name,
            kind="class",
            location=Location(self.file_path, node.lineno, _col(node), end_l, end_c),
            docstring=ast.get_docstring(node),
            parent=self._class_stack[-1] if self._class_stack else None,
            is_private=node.name.startswith("_"),
        ))
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        is_private = node.name.startswith("_") and node.name not in ("__init__", "__call__")
        if is_private and not self.include_private:
            return

        parent = self._class_stack[-1] if self._class_stack else None
        kind = ("async_method" if isinstance(node, ast.AsyncFunctionDef) else "method") \
               if parent else \
               ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")

        end_l, end_c = _end(node)
        self.symbols.append(SymbolDef(
            name=node.name,
            kind=kind,
            location=Location(self.file_path, node.lineno, _col(node), end_l, end_c),
            docstring=ast.get_docstring(node),
            signature=_sig(node),
            parent=parent,
            is_private=is_private,
        ))

        self.metrics.append(FunctionMetrics(
            name=node.name,
            location=Location(self.file_path, node.lineno, _col(node), end_l, end_c),
            loc=_loc(node, self.source_lines),
            cyclomatic_complexity=_cyclomatic(node),
            param_count=_param_count(node),
            nesting_depth=_nesting_depth(node),
            return_count=sum(1 for n in ast.walk(node) if isinstance(n, ast.Return)),
            parent_class=parent,
        ))

        self._class_stack.append(f"{parent}.{node.name}" if parent else node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(ImportEdge(
                from_file=self.file_path,
                module=alias.name,
                names=[],
                alias=alias.asname,
                line=node.lineno,
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(ImportEdge(
                from_file=self.file_path,
                module=node.module,
                names=[alias.name for alias in node.names],
                alias=None,
                line=node.lineno,
            ))


# ---------------------------------------------------------------------------
# Usage finder
# ---------------------------------------------------------------------------

class _UsageFinder(ast.NodeVisitor):
    def __init__(self, symbol_name: str, file_path: str, source_lines: list[str]):
        self.symbol_name = symbol_name
        self.file_path = file_path
        self.source_lines = source_lines
        self.usages: list[Usage] = []

    def _add(self, node: ast.AST):
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0) + 1
        ctx_line = self.source_lines[line - 1].rstrip() if 0 < line <= len(self.source_lines) else ""
        self.usages.append(Usage(
            symbol_name=self.symbol_name,
            location=Location(self.file_path, line, col),
            context_line=ctx_line,
        ))

    def visit_Name(self, node: ast.Name):
        if node.id == self.symbol_name:
            self._add(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr == self.symbol_name:
            self._add(node)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CodeIntelligence:
    """Pure-Python AST-based code intelligence. No third-party deps."""

    def __init__(self, include_private: bool = False, max_file_size_kb: int = 500):
        self.include_private = include_private
        self.max_file_size_kb = max_file_size_kb
        self._cache: dict[str, FileIntelligence] = {}

    def analyze_file(self, file_path: str) -> FileIntelligence:
        abs_path = str(Path(file_path).resolve())
        if abs_path in self._cache:
            return self._cache[abs_path]

        result = FileIntelligence(file=abs_path)

        try:
            size_kb = Path(abs_path).stat().st_size / 1024
            if size_kb > self.max_file_size_kb:
                result.parse_error = f"File too large ({size_kb:.0f} KB > {self.max_file_size_kb} KB)"
                return result

            source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=abs_path)
        except SyntaxError as e:
            result.parse_error = f"SyntaxError: {e}"
            return result
        except Exception as e:
            result.parse_error = str(e)
            return result

        visitor = _FileVisitor(abs_path, source, self.include_private)
        visitor.visit(tree)

        result.symbols = visitor.symbols
        result.imports = visitor.imports
        result.metrics = visitor.metrics
        self._cache[abs_path] = result
        return result

    def analyze_project(
        self,
        directory: str,
        ignore_dirs: list[str] | None = None,
    ) -> dict[str, FileIntelligence]:
        """Analyze all .py files under directory. Returns {abs_path: FileIntelligence}."""
        ignore = set(ignore_dirs or [".git", "__pycache__", "venv", ".venv"])
        results: dict[str, FileIntelligence] = {}
        root = Path(directory).resolve()

        for py_file in sorted(root.rglob("*.py")):
            if any(part in ignore for part in py_file.parts):
                continue
            intel = self.analyze_file(str(py_file))
            results[str(py_file)] = intel

        return results

    def lookup_symbol(
        self,
        name: str,
        project_intel: dict[str, FileIntelligence],
    ) -> list[SymbolDef]:
        """Find all definitions of a symbol across the project."""
        results = []
        for intel in project_intel.values():
            for sym in intel.symbols:
                if sym.name == name or (sym.parent and f"{sym.parent}.{sym.name}" == name):
                    results.append(sym)
        return sorted(results, key=lambda s: (s.location.file, s.location.line))

    def find_usages(
        self,
        symbol_name: str,
        project_intel: dict[str, FileIntelligence],
    ) -> list[Usage]:
        """Find all name/attribute usages of symbol_name across the project."""
        results: list[Usage] = []
        for abs_path, intel in project_intel.items():
            try:
                source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=abs_path)
                finder = _UsageFinder(symbol_name, abs_path, source.splitlines())
                finder.visit(tree)
                results.extend(finder.usages)
            except Exception:
                pass
        return sorted(results, key=lambda u: (u.location.file, u.location.line))

    def get_function_metrics(
        self,
        project_intel: dict[str, FileIntelligence],
        sort_by: str = "cyclomatic_complexity",
        top_n: int | None = None,
    ) -> list[FunctionMetrics]:
        """Return all FunctionMetrics sorted by sort_by descending."""
        all_metrics: list[FunctionMetrics] = []
        for intel in project_intel.values():
            all_metrics.extend(intel.metrics)
        all_metrics.sort(key=lambda m: getattr(m, sort_by, 0), reverse=True)
        return all_metrics[:top_n] if top_n else all_metrics

    def build_import_graph(
        self,
        project_intel: dict[str, FileIntelligence],
    ) -> dict[str, list[ImportEdge]]:
        """Return {file_path: [ImportEdge, ...]} for the whole project."""
        return {path: intel.imports for path, intel in project_intel.items()}

    def project_summary(
        self,
        project_intel: dict[str, FileIntelligence],
        project_root: str,
        top_n: int = 15,
    ) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the project."""
        all_metrics = self.get_function_metrics(project_intel, top_n=top_n)
        root = project_root

        def rel(p: str) -> str:
            return os.path.relpath(p, root)

        total_symbols = sum(len(i.symbols) for i in project_intel.values())
        total_functions = sum(
            1 for i in project_intel.values()
            for s in i.symbols if s.kind in ("function", "async_function", "method", "async_method")
        )
        total_classes = sum(
            1 for i in project_intel.values() for s in i.symbols if s.kind == "class"
        )
        parse_errors = {
            rel(p): i.parse_error
            for p, i in project_intel.items() if i.parse_error
        }

        return {
            "files_analyzed": len(project_intel),
            "parse_errors": parse_errors,
            "total_symbols": total_symbols,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "complexity_hotspots": [
                {
                    "name": m.name,
                    "parent_class": m.parent_class,
                    "file": rel(m.location.file),
                    "line": m.location.line,
                    "col": m.location.col,
                    "cyclomatic_complexity": m.cyclomatic_complexity,
                    "loc": m.loc,
                    "param_count": m.param_count,
                    "nesting_depth": m.nesting_depth,
                }
                for m in all_metrics
            ],
            "large_files": sorted(
                [
                    {"file": rel(p), "symbols": len(i.symbols), "functions": len(i.metrics)}
                    for p, i in project_intel.items()
                ],
                key=lambda x: x["functions"], reverse=True
            )[:top_n],
        }
