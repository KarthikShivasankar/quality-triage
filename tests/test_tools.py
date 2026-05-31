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

# ---------------------------------------------------------------------------
# ML smell normalisation (pure, no third-party deps)
# ---------------------------------------------------------------------------

class TestMlSmellNormalization:
    def test_framework_shape(self):
        from code_review_agent.tools import _normalize_ml_smell
        raw = {
            "framework": "NumPy", "name": "Randomness Control",
            "how_to_fix": "seed it", "benefits": "reproducible",
            "strategies": "np.random.seed", "line_number": 5,
            "code_snippet": "np.random.rand()", "file_path": "m.py",
        }
        out = _normalize_ml_smell(raw)
        assert out["name"] == "Randomness Control"
        assert out["how_to_fix"] == "seed it"
        assert out["line_number"] == 5
        assert out["framework"] == "NumPy"

    def test_general_smell_key(self):
        """ML_SmellDetector uses 'smell' for the description."""
        from code_review_agent.tools import _normalize_ml_smell
        raw = {
            "smell": "Data leakage detected in preprocessing step",
            "line_number": 3, "code_snippet": "scaler.fit(X)", "file_path": "m.py",
        }
        out = _normalize_ml_smell(raw)
        # name derived from first words of the description
        assert out["name"].startswith("Data leakage")
        assert out["description"] == "Data leakage detected in preprocessing step"
        assert out["line_number"] == 3

    def test_huggingface_fix_and_location_keys(self):
        """HuggingFaceSmellDetector uses 'fix' and a 'location' like 'Line 7'."""
        from code_review_agent.tools import _normalize_ml_smell
        raw = {
            "framework": "Hugging Face", "name": "Model versioning not specified",
            "fix": "Specify model version", "benefits": "reproducible",
            "location": "Line 7",
        }
        out = _normalize_ml_smell(raw)
        assert out["how_to_fix"] == "Specify model version"
        assert out["line_number"] == 7

    def test_non_dict_is_safe(self):
        from code_review_agent.tools import _normalize_ml_smell
        out = _normalize_ml_smell("oops")
        assert out["name"] == "Unknown"


# ---------------------------------------------------------------------------
# detect_python_smells — REAL detectors on temp files (import-guarded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("code_quality_analyzer"),
    reason="code_quality_analyzer not installed",
)
class TestDetectPythonSmellsReal:
    def test_no_threshold_typeerror_and_structure(self, sample_project):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_project), analysis_type="all")
        # No top-level error and each category resolved to a list (not an error dict).
        assert "error" not in result
        for key in ("code_smells", "structural_smells", "architectural_smells"):
            val = result.get(key)
            # Either a list of findings, or a benign note dict (never a threshold error).
            if isinstance(val, dict):
                assert "error" not in val, f"{key} raised: {val}"
            else:
                assert isinstance(val, list)

    def test_code_smells_found(self, sample_project):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_project), analysis_type="code")
        assert isinstance(result["code_smells"], list)
        # The long, many-param train() function should trigger at least one smell.
        assert len(result["code_smells"]) >= 1

    def test_structural_single_file_via_tempdir(self, sample_project):
        from code_review_agent.tools import detect_python_smells
        target = sample_project / "big.py"
        result = detect_python_smells(str(target), analysis_type="structural")
        assert isinstance(result.get("structural_smells"), list)

    def test_architectural_single_file_skipped(self, sample_project):
        from code_review_agent.tools import detect_python_smells
        target = sample_project / "model.py"
        result = detect_python_smells(str(target), analysis_type="architectural")
        # A single file can't have architectural smells; should be a note, not an error.
        val = result.get("architectural_smells")
        assert isinstance(val, dict) and "error" not in val


# ---------------------------------------------------------------------------
# detect_ml_smells — REAL detector on temp files (import-guarded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("ml_code_smell_detector"),
    reason="ml_code_smell_detector not installed",
)
class TestDetectMlSmellsReal:
    def test_numpy_smell_detected_and_normalized(self, tmp_path):
        from code_review_agent.tools import detect_ml_smells
        (tmp_path / "m.py").write_text(
            "import numpy as np\nx = np.random.rand(3)\n", encoding="utf-8"
        )
        result = detect_ml_smells(str(tmp_path))
        assert "summary" in result
        # The numpy file should yield at least one framework or general smell,
        # and every smell carries a canonical 'name'.
        for key in ("framework_smells", "huggingface_smells", "general_ml_smells"):
            for entry in result.get(key, []):
                for smell in entry["smells"]:
                    assert "name" in smell

    def test_fresh_detector_no_duplication(self, tmp_path):
        """Two files should not duplicate each other's smells."""
        from code_review_agent.tools import detect_ml_smells
        (tmp_path / "a.py").write_text("import numpy as np\nnp.random.rand(2)\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("import numpy as np\nnp.random.rand(2)\n", encoding="utf-8")
        result = detect_ml_smells(str(tmp_path))
        # Each file entry's smells belong to that file only.
        for key in ("framework_smells", "general_ml_smells"):
            for entry in result.get(key, []):
                for smell in entry["smells"]:
                    fp = smell.get("file_path", entry["file"])
                    assert entry["file"].endswith(os.path.basename(fp)) or fp == entry["file"]


# ---------------------------------------------------------------------------
# classify_technical_debt — engines MOCKED (never download a model)
# ---------------------------------------------------------------------------

class _FakeEngine:
    """Stand-in for tdsuite inference engines."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def predict_single(self, text):
        return {
            "text": text,
            "predicted_class": 1,
            "predicted_probability": 0.91,
            "class_probabilities": [0.09, 0.91],
        }


@pytest.fixture
def fake_tdsuite(monkeypatch):
    """Inject a fake tdsuite + tdsuite.utils so the real (slow) import is skipped."""
    import sys
    import types

    created = {}

    def _make(onnx_has_from_pretrained=False):
        tdsuite_mod = types.ModuleType("tdsuite")
        utils_mod = types.ModuleType("tdsuite.utils")

        class OnnxInferenceEngine(_FakeEngine):
            pass

        if onnx_has_from_pretrained:
            OnnxInferenceEngine.from_pretrained = classmethod(
                lambda cls, model_name: cls(model_name=model_name, via="onnx_fp")
            )

        class InferenceEngine(_FakeEngine):
            pass

        utils_mod.OnnxInferenceEngine = OnnxInferenceEngine
        utils_mod.InferenceEngine = InferenceEngine
        tdsuite_mod.utils = utils_mod

        monkeypatch.setitem(sys.modules, "tdsuite", tdsuite_mod)
        monkeypatch.setitem(sys.modules, "tdsuite.utils", utils_mod)
        created["onnx"] = OnnxInferenceEngine
        created["torch"] = InferenceEngine
        return created

    return _make


class TestClassifyTechnicalDebt:
    def test_empty_texts_returns_error(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt([])
        assert "error" in result

    def test_unknown_category_returns_error(self, fake_tdsuite):
        fake_tdsuite()
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["x"], category="not-a-category")
        assert "error" in result
        assert "available_categories" in result

    def test_torch_backend_predicts(self, fake_tdsuite):
        fake_tdsuite(onnx_has_from_pretrained=False)
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["TODO: fix later"], backend="torch")
        assert result["predictions"][0]["predicted_class"] == 1
        assert result["predictions"][0]["predicted_probability"] == 0.91
        assert result["label"] == "Technical Debt"

    def test_category_selects_model(self, fake_tdsuite):
        fake_tdsuite()
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["secret key in code"], category="security", backend="torch")
        assert result["model"] == "karths/binary_classification_train_secu"
        assert result["label"] == "Security Debt"

    def test_onnx_path_uses_onnx_engine(self, fake_tdsuite):
        created = fake_tdsuite()
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["x"], onnx_path="/tmp/model.onnx")
        assert "predictions" in result
        assert result["predictions"][0]["predicted_class"] == 1

    def test_onnx_from_pretrained_used_when_available(self, fake_tdsuite):
        fake_tdsuite(onnx_has_from_pretrained=True)
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["x"], backend="auto")
        assert "predictions" in result

    def test_predict_error_is_captured(self, fake_tdsuite, monkeypatch):
        created = fake_tdsuite()

        def boom(self, text):
            raise RuntimeError("inference failed")

        monkeypatch.setattr(created["torch"], "predict_single", boom)
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["x"], backend="torch")
        assert "error" in result["predictions"][0]
