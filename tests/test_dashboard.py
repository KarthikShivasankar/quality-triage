"""Dashboard HTML and table helpers."""

from __future__ import annotations

from code_review_agent.dashboard import (
    FINDINGS_HEADERS,
    findings_table_rows,
    print_review_dashboard,
    render_html_report,
)
from code_review_agent.reporter import Finding, ReportData, Severity, build_report


def test_findings_table_rows():
    data = ReportData(
        target="/p",
        analyzed_at="now",
        provider="ollama",
        model="x",
        files_analyzed=1,
        findings=[
            Finding(
                "MLFW-001",
                "ml_smells",
                "data_leakage",
                Severity.CRITICAL,
                "train.py",
                "/p/train.py",
                12,
                4,
                message="fit on test",
            ),
        ],
    )
    rows = findings_table_rows(data)
    assert FINDINGS_HEADERS[0] == "Severity"
    assert rows[0][0] == "CRITICAL"
    assert "train.py:12:4" in rows[0][2]


def test_html_report_empty_and_with_narrative():
    empty = build_report(target="/proj", provider="ollama", model="x")
    html = render_html_report(empty)
    assert "No findings" in html
    empty.narrative = "All clear."
    html2 = render_html_report(empty)
    assert "All clear." in html2


def test_print_review_dashboard_does_not_crash():
    from rich.console import Console

    data = build_report(target="/proj", provider="ollama", model="x")
    data.narrative = "ok"
    console = Console(record=True, width=80)
    print_review_dashboard(data, console)
    text = console.export_text()
    assert "Quality Triage" in text
    assert "Health" in text


def test_findings_rows_from_payload():
    from code_review_agent.dashboard import findings_rows_from_payload

    rows = findings_rows_from_payload(
        {
            "findings": [
                {
                    "severity": "high",
                    "finding_id": "PY-1",
                    "file": "a.py",
                    "line": 3,
                    "col": 2,
                    "category": "long_method",
                    "symbol": "run",
                    "message": "too long",
                }
            ]
        }
    )
    assert rows[0][0] == "HIGH"
    assert "a.py:3:2" in rows[0][2]
