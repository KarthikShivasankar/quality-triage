"""Direct tests for AST code intelligence."""

from __future__ import annotations

from code_review_agent.code_intel import CodeIntelligence


def _write_sample(tmp_path):
    (tmp_path / "mod.py").write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "class Worker:\n"
        "    def run(self, x, y):\n"
        "        if x:\n"
        "            if y:\n"
        "                return x + y\n"
        "        return 0\n\n"
        "def helper(a):\n"
        "    return os.path.join('a', str(a))\n"
        "helper(1)\n"
    )
    return tmp_path / "mod.py"


def test_analyze_file_symbols_and_metrics(tmp_path):
    path = _write_sample(tmp_path)
    ci = CodeIntelligence()
    intel = ci.analyze_file(str(path))
    names = {s.name for s in intel.symbols}
    assert "Worker" in names
    assert "run" in names
    assert "helper" in names
    run = next(m for m in intel.metrics if m.name == "run")
    assert run.cyclomatic_complexity >= 3
    assert run.nesting_depth >= 2
    assert run.parent_class == "Worker"


def test_lookup_and_usages(tmp_path):
    path = _write_sample(tmp_path)
    ci = CodeIntelligence()
    imap = {str(path): ci.analyze_file(str(path))}
    defs = ci.lookup_symbol("Worker", imap)
    assert defs and defs[0].kind == "class"
    usages = ci.find_usages("helper", imap)
    assert usages


def test_project_summary(tmp_path):
    _write_sample(tmp_path)
    ci = CodeIntelligence()
    imap = ci.analyze_project(str(tmp_path), ignore_dirs=[])
    summary = ci.project_summary(imap, str(tmp_path), top_n=5)
    assert summary["files_analyzed"] >= 1
    assert summary["total_classes"] >= 1
    assert summary["complexity_hotspots"]


def test_parse_error_recorded(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def (\n")
    ci = CodeIntelligence()
    intel = ci.analyze_file(str(bad))
    assert intel.parse_error
