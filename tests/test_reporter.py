"""
Tests for code_review_agent.reporter — offline, pure functions.
"""

from __future__ import annotations

import json

from code_review_agent.reporter import (
    FindingNormalizer,
    ReportRenderer,
    build_report,
    save_report,
)


# ---------------------------------------------------------------------------
# Binary TD normalisation
# ---------------------------------------------------------------------------

class TestTdNormalization:
    def test_class_1_kept_and_labeled(self):
        norm = FindingNormalizer("/root")
        raw = {
            "label": "Security Debt",
            "predictions": [
                {"text": "hardcoded password", "predicted_class": 1, "predicted_probability": 0.88},
            ],
        }
        out = norm.normalize_td_predictions(raw)
        assert len(out) == 1
        assert out[0]["category"] == "Security Debt"
        assert out[0]["confidence"] == 0.88

    def test_class_0_skipped(self):
        norm = FindingNormalizer("/root")
        raw = {
            "label": "Technical Debt",
            "predictions": [
                {"text": "clean code", "predicted_class": 0, "predicted_probability": 0.95},
            ],
        }
        assert norm.normalize_td_predictions(raw) == []

    def test_default_label_when_missing(self):
        norm = FindingNormalizer("/root")
        raw = {"predictions": [{"text": "x", "predicted_class": 1, "predicted_probability": 0.7}]}
        out = norm.normalize_td_predictions(raw)
        assert out[0]["category"] == "Technical Debt"

    def test_error_prediction_captured(self):
        norm = FindingNormalizer("/root")
        raw = {"label": "Code Debt", "predictions": [{"text": "x", "error": "boom"}]}
        out = norm.normalize_td_predictions(raw)
        assert len(out) == 1
        assert out[0]["error"] == "boom"

    def test_mixed_predictions(self):
        norm = FindingNormalizer("/root")
        raw = {
            "label": "Test Debt",
            "predictions": [
                {"text": "a", "predicted_class": 1, "predicted_probability": 0.6},
                {"text": "b", "predicted_class": 0, "predicted_probability": 0.99},
                {"text": "c", "predicted_class": 1, "predicted_probability": 0.75},
            ],
        }
        out = norm.normalize_td_predictions(raw)
        assert len(out) == 2
        assert all(p["category"] == "Test Debt" for p in out)


# ---------------------------------------------------------------------------
# ML smell normalisation (both key shapes)
# ---------------------------------------------------------------------------

class TestMlNormalization:
    def test_name_shape(self):
        norm = FindingNormalizer("/root")
        raw = {
            "framework_smells": [
                {"file": "/root/m.py", "smells": [
                    {"name": "Data Leakage", "framework": "sklearn",
                     "how_to_fix": "split first", "line_number": 10},
                ]},
            ]
        }
        findings = norm.normalize_ml_smells(raw)
        assert len(findings) == 1
        assert findings[0].category == "Data Leakage"
        assert findings[0].line == 10
        assert findings[0].framework == "sklearn"

    def test_smell_key_shape(self):
        norm = FindingNormalizer("/root")
        raw = {
            "general_ml_smells": [
                {"file": "/root/m.py", "smells": [
                    {"smell": "Missing random seed for reproducibility", "line_number": 4},
                ]},
            ]
        }
        findings = norm.normalize_ml_smells(raw)
        assert len(findings) == 1
        assert findings[0].message == "Missing random seed for reproducibility"
        assert findings[0].line == 4

    def test_location_string_shape(self):
        norm = FindingNormalizer("/root")
        raw = {
            "huggingface_smells": [
                {"file": "/root/m.py", "smells": [
                    {"name": "Model versioning", "fix": "pin version", "location": "Line 12"},
                ]},
            ]
        }
        findings = norm.normalize_ml_smells(raw)
        assert findings[0].line == 12
        assert findings[0].how_to_fix == "pin version"


# ---------------------------------------------------------------------------
# Python smell normalisation
# ---------------------------------------------------------------------------

class TestPythonNormalization:
    def test_code_smells_list(self):
        norm = FindingNormalizer("/root")
        raw = {
            "code_smells": [
                {"name": "Long Method", "file_path": "/root/a.py", "line_number": 5,
                 "severity": "high", "description": "too long", "module_class": "Foo.bar"},
            ]
        }
        findings = norm.normalize_python_smells(raw)
        assert len(findings) == 1
        assert findings[0].category == "Long Method"
        assert findings[0].severity.value == "high"
        assert findings[0].symbol == "Foo.bar"


# ---------------------------------------------------------------------------
# Renderers (smoke)
# ---------------------------------------------------------------------------

class TestRenderers:
    def _report(self):
        ml_raw = {
            "framework_smells": [
                {"file": "/root/m.py", "smells": [
                    {"name": "Data Leakage", "framework": "sklearn",
                     "how_to_fix": "split", "line_number": 10,
                     "code_snippet": "scaler.fit(X)"},
                ]},
            ]
        }
        py_raw = {
            "code_smells": [
                {"name": "Long Method", "file_path": "/root/a.py", "line_number": 5,
                 "severity": "high", "description": "too long"},
            ]
        }
        td_raw = {
            "label": "Security Debt",
            "predictions": [
                {"text": "TODO: hardcoded secret", "predicted_class": 1, "predicted_probability": 0.9},
            ],
        }
        return build_report(
            target="/root", provider="ollama", model="qwen2.5-coder:7b",
            ml_raw=ml_raw, py_raw=py_raw, td_raw=td_raw,
            files_analyzed=2, tools_run=["ml", "py", "td"],
        )

    def test_markdown_render(self):
        data = self._report()
        md = ReportRenderer().render_markdown(data)
        assert "# Code Review Report" in md
        assert "Data Leakage" in md
        assert "Long Method" in md
        assert "Security Debt" in md

    def test_json_render(self):
        data = self._report()
        out = ReportRenderer().render_json(data)
        parsed = json.loads(out)
        assert parsed["provider"] == "ollama"
        assert len(parsed["findings"]) == 2
        assert parsed["td_predictions"][0]["category"] == "Security Debt"

    def test_save_report_both(self, tmp_path):
        data = self._report()
        written = save_report(data, output_dir=str(tmp_path), fmt="both")
        assert len(written) == 2
        assert any(p.endswith(".md") for p in written)
        assert any(p.endswith(".json") for p in written)
