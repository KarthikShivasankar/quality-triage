"""Tests for smell-family and finding filters."""

from __future__ import annotations

from code_review_agent.filters import (
    filter_findings,
    filter_td_predictions,
    normalize_smell_families,
    render_filtered_summary,
    split_smell_families,
)
from code_review_agent.reporter import Finding, Severity, build_report


def _finding(fid: str, category: str, tool: str = "python_smells") -> Finding:
    prefix = fid.split("-")[0]
    fam_map = {"PYCS": "code", "PYAS": "architectural", "MLFW": "framework"}
    return Finding(
        finding_id=fid,
        tool=tool,
        category=category,
        severity=Severity.HIGH,
        file="a.py",
        file_abs="/proj/a.py",
        line=1,
        col=None,
        message=f"msg for {category}",
        extra={"smell_family": fam_map.get(prefix, "code")},
    )


def test_normalize_and_split_smell_families():
    all_fams = normalize_smell_families(None)
    assert "code" in all_fams and "general" in all_fams
    assert normalize_smell_families(["code", "bogus", "general"]) == ["code", "general"]
    py, ml = split_smell_families(["code", "framework"])
    assert py == ["code"]
    assert ml == ["framework"]


def test_filter_findings_by_family_and_category():
    findings = [
        _finding("PYCS-001", "Long Method"),
        _finding("PYAS-001", "God Object"),
        _finding("MLFW-001", "Missing Seed", tool="ml_smells"),
    ]
    out = filter_findings(findings, families=["code"], categories=["Long Method"])
    assert len(out) == 1
    assert out[0].category == "Long Method"


def test_filter_findings_search():
    findings = [_finding("PYCS-001", "Long Method")]
    assert filter_findings(findings, search="long method")
    assert not filter_findings(findings, search="nope")


def test_filter_td_predictions():
    preds = [
        {"text": "#12 Fix parser", "category": "Code Debt", "confidence": 0.9},
        {"text": "TODO: x", "category": "No Debt", "confidence": 0.2},
    ]
    out = filter_td_predictions(preds, search="parser")
    assert len(out) == 1


def test_render_filtered_summary():
    data = build_report(
        target="/proj",
        provider="ollama",
        model="x",
        py_raw={
            "code_smells": [
                {
                    "name": "Long Method",
                    "file_path": "/proj/a.py",
                    "line_number": 1,
                    "severity": "high",
                    "description": "too long",
                }
            ]
        },
    )
    payload = {
        "target": data.target,
        "health_score": data.health_score,
        "findings": [f.__dict__ for f in data.findings],
        "td_predictions": [],
    }
    md = render_filtered_summary(payload, data.findings)
    assert "Filtered findings" in md
    assert "Long Method" in md
