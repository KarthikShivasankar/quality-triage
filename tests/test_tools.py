"""
Tests for code_review_agent.tools (pure-Python helpers only).

Tests that require optional third-party detectors (ml_code_smell_detector,
code_quality_analyzer, tdsuite) are skipped gracefully when the packages
are not installed, keeping CI green on a minimal environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_review_agent.config import reset_config


@pytest.fixture(autouse=True)
def reset_cfg():
    """Ensure config singleton is cleared between tests."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Helpers: _rel, _enrich_column, _python_files
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_rel_same_root(self, tmp_path):
        from code_review_agent.tools import _rel
        child = str(tmp_path / "a" / "b.py")
        result = _rel(child, str(tmp_path))
        assert result == os.path.join("a", "b.py")

    def test_rel_fallback_on_different_drive(self, tmp_path, monkeypatch):
        """_rel should return abs_path when relpath raises ValueError."""
        from code_review_agent.tools import _rel

        original = os.path.relpath

        def mock_relpath(p, start):
            raise ValueError("different drive")

        monkeypatch.setattr(os.path, "relpath", mock_relpath)
        result = _rel("/some/abs/path.py", "/other/root")
        assert result == "/some/abs/path.py"

    def test_enrich_column_finds_needle(self, tmp_path):
        from code_review_agent.tools import _enrich_column
        f = tmp_path / "sample.py"
        f.write_text("x = 1\ndef foo(): pass\n")
        col = _enrich_column(str(f), 2, "def foo():")
        assert col == 1  # column of "def" on line 2

    def test_enrich_column_missing_line(self, tmp_path):
        from code_review_agent.tools import _enrich_column
        f = tmp_path / "sample.py"
        f.write_text("x = 1\n")
        col = _enrich_column(str(f), 99, "anything")
        assert col is None

    def test_enrich_column_nonexistent_file(self):
        from code_review_agent.tools import _enrich_column
        col = _enrich_column("/nonexistent/file.py", 1, "x")
        assert col is None

    def test_python_files_single_file(self, tmp_path):
        from code_review_agent.tools import _python_files
        f = tmp_path / "module.py"
        f.write_text("x = 1")
        result = _python_files(f, set())
        assert result == [f]

    def test_python_files_directory(self, tmp_path):
        from code_review_agent.tools import _python_files
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("")
        result = _python_files(tmp_path, set())
        paths = {p.name for p in result}
        assert paths == {"a.py", "b.py", "c.py"}

    def test_python_files_ignores_dirs(self, tmp_path):
        from code_review_agent.tools import _python_files
        (tmp_path / "a.py").write_text("")
        ignored = tmp_path / "__pycache__"
        ignored.mkdir()
        (ignored / "b.py").write_text("")
        result = _python_files(tmp_path, {"__pycache__"})
        names = [p.name for p in result]
        assert "b.py" not in names
        assert "a.py" in names


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "example.py"
        f.write_text("line1\nline2\nline3\n")
        result = read_file(str(f))
        assert result["total_lines"] == 3
        assert result["shown_lines"] == 3
        assert result["truncated"] is False
        assert "line1" in result["content"]
        assert "line2" in result["content"]

    def test_truncates_to_max_lines(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "long.py"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        result = read_file(str(f), max_lines=10)
        assert result["shown_lines"] == 10
        assert result["total_lines"] == 100
        assert result["truncated"] is True

    def test_missing_file_returns_error(self, tmp_path):
        from code_review_agent.tools import read_file
        result = read_file(str(tmp_path / "ghost.py"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_directory_returns_error(self, tmp_path):
        from code_review_agent.tools import read_file
        result = read_file(str(tmp_path))
        assert "error" in result

    def test_line_numbers_in_content(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "numbered.py"
        f.write_text("a = 1\nb = 2\n")
        result = read_file(str(f))
        assert "1 |" in result["content"] or "   1 |" in result["content"]


# ---------------------------------------------------------------------------
# list_python_files
# ---------------------------------------------------------------------------

class TestListPythonFiles:
    def test_lists_files_in_dir(self, tmp_path):
        from code_review_agent.tools import list_python_files
        (tmp_path / "foo.py").write_text("x=1")
        (tmp_path / "bar.py").write_text("y=2")
        result = list_python_files(str(tmp_path))
        assert result["total_files"] == 2
        names = {f["path"] for f in result["files"]}
        assert "foo.py" in names
        assert "bar.py" in names

    def test_excludes_ignored_dirs(self, tmp_path):
        from code_review_agent.tools import list_python_files
        (tmp_path / "main.py").write_text("")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "hidden.py").write_text("")
        result = list_python_files(str(tmp_path), ignore_dirs=[".venv"])
        names = {f["path"] for f in result["files"]}
        assert "main.py" in names
        assert not any(".venv" in p for p in names)

    def test_missing_dir_returns_error(self, tmp_path):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(tmp_path / "does_not_exist"))
        assert "error" in result

    def test_file_path_returns_error(self, tmp_path):
        from code_review_agent.tools import list_python_files
        f = tmp_path / "file.py"
        f.write_text("")
        result = list_python_files(str(f))
        assert "error" in result

    def test_file_size_reported(self, tmp_path):
        from code_review_agent.tools import list_python_files
        f = tmp_path / "sized.py"
        f.write_text("x" * 1024)
        result = list_python_files(str(tmp_path))
        entry = next(e for e in result["files"] if "sized.py" in e["path"])
        assert entry["size_bytes"] == 1024
        assert entry["size_kb"] == 1.0

    def test_empty_dir_returns_zero_files(self, tmp_path):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(tmp_path))
        assert result["total_files"] == 0
        assert result["files"] == []


# ---------------------------------------------------------------------------
# detect_ml_smells — import-guarded
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("ml_code_smell_detector"),
    reason="ml_code_smell_detector not installed",
)
class TestDetectMlSmells:
    def test_nonexistent_path_returns_error(self):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells("/nonexistent/path")
        assert "error" in result

    def test_empty_dir_returns_error(self, tmp_path):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells(str(tmp_path))
        assert "error" in result

    def test_returns_summary_keys(self, tmp_path):
        from code_review_agent.tools import detect_ml_smells
        (tmp_path / "model.py").write_text("import numpy as np\nx = np.array([1,2,3])\n")
        result = detect_ml_smells(str(tmp_path))
        if "error" not in result:
            assert "summary" in result
            assert "files_analyzed" in result["summary"]


# ---------------------------------------------------------------------------
# classify_technical_debt — import-guarded
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tdsuite"),
    reason="tdsuite not installed",
)
class TestClassifyTechnicalDebt:
    def test_empty_texts_returns_error(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt([])
        assert "error" in result

    def test_returns_predictions_key(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["TODO: fix this later"])
        if "error" not in result:
            assert "predictions" in result
            assert len(result["predictions"]) == 1
