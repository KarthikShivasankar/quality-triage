"""
Tests for code_review_agent.code_intel — pure AST analysis, fully offline.
"""

from __future__ import annotations

import pytest

from code_review_agent.code_intel import CodeIntelligence


SAMPLE = '''\
import os
from collections import defaultdict


def top_level(a, b, c):
    if a:
        for i in range(b):
            if i > c:
                return i
    return 0


class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self, loud=False):
        msg = "hi " + self.name
        if loud:
            return msg.upper()
        return msg
'''


@pytest.fixture
def project(tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "from a import Greeter\n\n\ndef use():\n    g = Greeter('x')\n    return g.greet()\n",
        encoding="utf-8",
    )
    return tmp_path


class TestSymbols:
    def test_finds_functions_and_classes(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        defs = ci.lookup_symbol("Greeter", intel)
        assert len(defs) == 1
        assert defs[0].kind == "class"

    def test_method_signature(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        defs = ci.lookup_symbol("greet", intel)
        assert defs[0].kind == "method"
        assert "loud" in defs[0].signature
        assert defs[0].parent == "Greeter"


class TestMetrics:
    def test_cyclomatic_loc_params_nesting(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        metrics = ci.get_function_metrics(intel)
        top = next(m for m in metrics if m.name == "top_level")
        assert top.cyclomatic_complexity >= 4  # if + for + if + bool
        assert top.param_count == 3
        assert top.nesting_depth >= 3
        assert top.loc >= 4

    def test_self_excluded_from_param_count(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        metrics = ci.get_function_metrics(intel)
        greet = next(m for m in metrics if m.name == "greet")
        assert greet.param_count == 1  # 'self' excluded, 'loud' counted


class TestUsages:
    def test_find_usages_across_files(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        usages = ci.find_usages("Greeter", intel)
        files = {u.location.file for u in usages}
        # The Greeter() call in b.py is a Name usage and must be found.
        assert len(usages) >= 1
        assert any(f.endswith("b.py") for f in files)


class TestImportGraph:
    def test_build_import_graph(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        graph = ci.build_import_graph(intel)
        b_file = next(p for p in graph if p.endswith("b.py"))
        modules = {e.module for e in graph[b_file]}
        assert "a" in modules


class TestProjectSummary:
    def test_summary_counts(self, project):
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(project))
        summary = ci.project_summary(intel, str(project), top_n=5)
        assert summary["files_analyzed"] == 2
        assert summary["total_classes"] >= 1
        assert summary["total_functions"] >= 3
        assert isinstance(summary["complexity_hotspots"], list)
        assert summary["complexity_hotspots"][0]["cyclomatic_complexity"] >= 1

    def test_parse_error_recorded(self, tmp_path):
        (tmp_path / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
        ci = CodeIntelligence()
        intel = ci.analyze_project(str(tmp_path))
        summary = ci.project_summary(intel, str(tmp_path))
        assert summary["parse_errors"]


class TestSingleFile:
    def test_analyze_single_file(self, project):
        ci = CodeIntelligence()
        result = ci.analyze_file(str(project / "a.py"))
        assert result.parse_error is None
        names = {s.name for s in result.symbols}
        assert "Greeter" in names
        assert "top_level" in names
