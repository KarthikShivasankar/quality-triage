"""Tests for structured reporting and SARIF."""

from __future__ import annotations

import json

from code_review_agent.reporter import (
    Finding,
    ReportRenderer,
    Severity,
    build_report,
    compute_health_score,
    count_by_severity,
    filter_by_min_severity,
    findings_meet_threshold,
    health_label,
    save_report,
)


def _ml_raw():
    return {
        "framework_smells": [
            {
                "file": "/proj/train.py",
                "smells": [
                    {
                        "name": "data_leakage",
                        "description": "fit on test",
                        "line_number": 12,
                        "how_to_fix": "split first",
                        "code_snippet": "model.fit(X)",
                    }
                ],
            }
        ],
        "huggingface_smells": [],
        "general_ml_smells": [],
    }


def _py_raw():
    return {
        "code_smells": [
            {
                "name": "Long Method",
                "file_path": "/proj/big.py",
                "line_number": 40,
                "severity": "high",
                "description": "too long",
                "module_class": "Worker.run",
            }
        ],
        "architectural_smells": [],
        "structural_smells": [],
    }


def test_build_report_normalises_and_sorts():
    data = build_report(
        target="/proj",
        provider="ollama",
        model="ollama/gemma3:latest",
        ml_raw=_ml_raw(),
        py_raw=_py_raw(),
        files_analyzed=2,
        tools_run=["detect_ml_smells"],
    )
    assert len(data.findings) == 2
    assert data.findings[0].severity is Severity.CRITICAL
    assert data.findings[0].category == "data_leakage"
    assert data.findings[1].symbol == "Worker.run"


def test_td_predictions_skip_no_debt():
    data = build_report(
        target="/proj",
        provider="ollama",
        model="x",
        td_raw={
            "predictions": [
                {"text": "ok", "predicted_class": 17, "predicted_probability": 0.9},
                {
                    "text": "TODO: leak",
                    "predicted_class": 14,
                    "predicted_probability": 0.8,
                },
            ]
        },
    )
    assert len(data.td_predictions) == 1
    assert data.td_predictions[0]["category"] == "Security Debt"


def test_td_class_label_binary_and_multiclass():
    from code_review_agent.reporter import td_class_label

    assert (
        td_class_label(
            {
                "predicted_class": 1,
                "class_probabilities": [0.01, 0.99],
            }
        )
        == "Technical Debt"
    )
    assert (
        td_class_label(
            {
                "predicted_class": 0,
                "class_probabilities": [0.99, 0.01],
            }
        )
        == "No Debt"
    )
    assert td_class_label({"predicted_class": 14}) == "Security Debt"


def test_td_binary_predictions_skip_no_debt():
    data = build_report(
        target="/proj",
        provider="ollama",
        model="x",
        td_raw={
            "predictions": [
                {
                    "text": "docs only",
                    "predicted_class": 0,
                    "predicted_probability": 0.99,
                    "class_probabilities": [0.99, 0.01],
                },
                {
                    "text": "FIXME: remove this hack",
                    "predicted_class": 1,
                    "predicted_probability": 0.98,
                    "class_probabilities": [0.02, 0.98],
                },
            ]
        },
    )
    assert len(data.td_predictions) == 1
    assert data.td_predictions[0]["category"] == "Technical Debt"


def test_render_markdown_includes_narrative():
    data = build_report(target="/proj", provider="ollama", model="x", ml_raw=_ml_raw())
    data.narrative = "Health score 40."
    md = ReportRenderer().render_markdown(data)
    assert "AI Synthesis" in md
    assert "Health score 40." in md
    assert "CRITICAL" in md


def test_render_json_and_sarif():
    data = build_report(target="/proj", provider="ollama", model="x", ml_raw=_ml_raw())
    renderer = ReportRenderer()
    payload = json.loads(renderer.render_json(data))
    assert payload["findings"]
    sarif = json.loads(renderer.render_sarif(data))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"][0]["level"] == "error"
    assert (
        "train.py"
        in sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
            "artifactLocation"
        ]["uri"]
    )


def test_filter_and_fail_on():
    findings = [
        Finding("a", "t", "c", Severity.LOW, "a.py", "/a.py", 1, None, message="x"),
        Finding("b", "t", "c", Severity.HIGH, "b.py", "/b.py", 2, None, message="y"),
    ]
    assert len(filter_by_min_severity(findings, "high")) == 1
    assert findings_meet_threshold(findings, "high") is True
    assert findings_meet_threshold(findings, "critical") is False
    assert findings_meet_threshold(findings, "none") is False


def test_save_report_sarif(tmp_path):
    data = build_report(
        target=str(tmp_path), provider="ollama", model="x", ml_raw=_ml_raw()
    )
    written = save_report(data, output_dir=str(tmp_path / "out"), fmt="sarif")
    path = __import__("pathlib").Path(written[0])
    assert path.suffix == ".sarif"
    doc = json.loads(path.read_text())
    assert doc["version"] == "2.1.0"


def test_health_score_and_label():
    clean = []
    assert compute_health_score(clean) == 100
    assert health_label(100) == "healthy"
    findings = [
        Finding(
            "a", "t", "c", Severity.CRITICAL, "a.py", "/a.py", 1, None, message="x"
        ),
        Finding("b", "t", "c", Severity.LOW, "b.py", "/b.py", 2, None, message="y"),
    ]
    assert compute_health_score(findings) == 81
    assert count_by_severity(findings)["critical"] == 1
    assert health_label(40) == "at risk"
    assert health_label(60) == "needs attention"


def test_render_markdown_empty_findings():
    data = build_report(target="/proj", provider="ollama", model="x")
    md = ReportRenderer().render_markdown(data)
    assert "Health score:" in md
    assert "No findings" in md


def test_render_json_includes_health():
    data = build_report(target="/proj", provider="ollama", model="x", ml_raw=_ml_raw())
    payload = json.loads(ReportRenderer().render_json(data))
    assert "health_score" in payload
    assert payload["severity_counts"]["critical"] == 1


def test_render_html_and_save(tmp_path):
    data = build_report(
        target=str(tmp_path), provider="ollama", model="x", ml_raw=_ml_raw()
    )
    html = ReportRenderer().render_html(data)
    assert "Quality Triage" in html
    assert "data_leakage" in html
    written = save_report(data, output_dir=str(tmp_path / "out"), fmt="html")
    assert written[0].endswith(".html")
    assert "inspection slip" in __import__("pathlib").Path(written[0]).read_text()


def test_ml_smells_dict_container():
    raw = {
        "framework_smells": [
            {
                "file": "/proj/a.py",
                "smells": {
                    "magic_number": [
                        {"name": "magic_number", "line_number": 3, "description": "n"}
                    ],
                },
            }
        ],
    }
    data = build_report(target="/proj", provider="ollama", model="x", ml_raw=raw)
    assert any(f.category == "magic_number" for f in data.findings)
