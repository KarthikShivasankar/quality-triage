"""
Gradio-based test report dashboard for Quality Triage.

Runs the comprehensive Ollama backend test suite via pytest and presents
results in a rich, interactive Gradio UI with:
  - Live progress log
  - Per-test pass/fail table
  - Summary metrics (pass rate, skipped, failed)
  - Expandable error details
  - Downloadable JSON report

Usage:
    uv run python tests/gradio_report.py
    uv run python tests/gradio_report.py --port 7861

The dashboard opens in your browser automatically.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── ensure src/ is importable ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

# ---------------------------------------------------------------------------
# Pytest runner with JSON output
# ---------------------------------------------------------------------------

PYTEST_JSON_OUT = ROOT / "reports" / "test_results.json"


def _run_tests(selected_markers: str) -> tuple[dict, str]:
    """
    Run pytest and return (parsed_json_report, raw_log_text).
    Uses pytest-json-report if available, otherwise falls back to parsing
    the verbose terminal output.
    """
    PYTEST_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        str(ROOT / "tests" / "test_ollama_backend.py"),
        "-v", "--tb=short", "--no-header",
        f"--json-report",
        f"--json-report-file={PYTEST_JSON_OUT}",
        "-p", "no:cacheprovider",
    ]

    # Add marker filter if specified
    if selected_markers and selected_markers.strip() and selected_markers.lower() != "all":
        cmd += ["-k", selected_markers.strip()]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        raw_log = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {}, "⏱ Pytest timed out after 10 minutes."
    except FileNotFoundError:
        return {}, "❌ pytest not found. Run: uv sync"

    # Try loading JSON report
    report: dict = {}
    if PYTEST_JSON_OUT.exists():
        try:
            with open(PYTEST_JSON_OUT) as f:
                report = json.load(f)
        except Exception:
            pass

    # Fallback: parse terminal output into minimal structure
    if not report:
        report = _parse_terminal_output(raw_log)

    return report, raw_log


def _parse_terminal_output(text: str) -> dict:
    """Minimal fallback parser for pytest -v output."""
    tests = []
    for line in text.splitlines():
        line = line.strip()
        for suffix, outcome in ((" PASSED", "passed"), (" FAILED", "failed"),
                                (" SKIPPED", "skipped"), (" ERROR", "error")):
            if line.endswith(suffix):
                node_id = line[: -len(suffix)].strip()
                tests.append({"nodeid": node_id, "outcome": outcome, "call": {}})
                break
    summary = {
        "passed": sum(1 for t in tests if t["outcome"] == "passed"),
        "failed": sum(1 for t in tests if t["outcome"] == "failed"),
        "skipped": sum(1 for t in tests if t["outcome"] == "skipped"),
        "error": sum(1 for t in tests if t["outcome"] == "error"),
        "total": len(tests),
    }
    return {"tests": tests, "summary": summary}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_summary_html(summary: dict, duration: float) -> str:
    total   = summary.get("total", 0)
    passed  = summary.get("passed", 0)
    failed  = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    error   = summary.get("error", 0)
    pass_rate = (passed / total * 100) if total else 0

    color = "#22c55e" if failed == 0 and error == 0 else "#ef4444"
    status_text = "ALL TESTS PASSED" if (failed == 0 and error == 0) else "SOME TESTS FAILED"

    return f"""
<div style="
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 2px solid {color};
    border-radius: 12px;
    padding: 24px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #cdd6f4;
    margin-bottom: 16px;
">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
    <span style="font-size:32px;">{"✅" if failed == 0 and error == 0 else "❌"}</span>
    <div>
      <div style="font-size:22px; font-weight:700; color:{color};">{status_text}</div>
      <div style="font-size:13px; color:#a6adc8;">Ran at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {duration:.1f}s total</div>
    </div>
  </div>

  <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:12px; text-align:center;">
    <div style="background:#313244; border-radius:8px; padding:12px;">
      <div style="font-size:28px; font-weight:700; color:#cba6f7;">{total}</div>
      <div style="font-size:12px; color:#a6adc8; text-transform:uppercase;">Total</div>
    </div>
    <div style="background:#313244; border-radius:8px; padding:12px;">
      <div style="font-size:28px; font-weight:700; color:#a6e3a1;">{passed}</div>
      <div style="font-size:12px; color:#a6adc8; text-transform:uppercase;">Passed</div>
    </div>
    <div style="background:#313244; border-radius:8px; padding:12px;">
      <div style="font-size:28px; font-weight:700; color:#f38ba8;">{failed + error}</div>
      <div style="font-size:12px; color:#a6adc8; text-transform:uppercase;">Failed</div>
    </div>
    <div style="background:#313244; border-radius:8px; padding:12px;">
      <div style="font-size:28px; font-weight:700; color:#f9e2af;">{skipped}</div>
      <div style="font-size:12px; color:#a6adc8; text-transform:uppercase;">Skipped</div>
    </div>
    <div style="background:#313244; border-radius:8px; padding:12px;">
      <div style="font-size:28px; font-weight:700; color:{"#a6e3a1" if pass_rate >= 80 else "#f38ba8"};">{pass_rate:.0f}%</div>
      <div style="font-size:12px; color:#a6adc8; text-transform:uppercase;">Pass Rate</div>
    </div>
  </div>
</div>
"""


def _outcome_badge(outcome: str) -> str:
    badges = {
        "passed":  ('<span style="background:#a6e3a1;color:#1e1e2e;border-radius:4px;'
                    'padding:2px 8px;font-size:11px;font-weight:700;">✓ PASS</span>'),
        "failed":  ('<span style="background:#f38ba8;color:#1e1e2e;border-radius:4px;'
                    'padding:2px 8px;font-size:11px;font-weight:700;">✗ FAIL</span>'),
        "skipped": ('<span style="background:#f9e2af;color:#1e1e2e;border-radius:4px;'
                    'padding:2px 8px;font-size:11px;font-weight:700;">⊘ SKIP</span>'),
        "error":   ('<span style="background:#fab387;color:#1e1e2e;border-radius:4px;'
                    'padding:2px 8px;font-size:11px;font-weight:700;">⚡ ERR</span>'),
    }
    return badges.get(outcome, outcome)


def _build_results_table(tests: list[dict]) -> str:
    if not tests:
        return "<p style='color:#a6adc8;'>No test results found.</p>"

    rows = []
    for i, t in enumerate(tests, 1):
        node_id = t.get("nodeid", "unknown")
        # Extract class and test name
        parts = node_id.split("::")
        if len(parts) >= 3:
            class_name = parts[-2]
            test_name  = parts[-1]
        elif len(parts) == 2:
            class_name = "—"
            test_name  = parts[-1]
        else:
            class_name = "—"
            test_name  = node_id

        outcome = t.get("outcome", "unknown")
        badge   = _outcome_badge(outcome)

        duration = ""
        call = t.get("call") or {}
        if isinstance(call, dict) and "duration" in call:
            duration = f"{call['duration']:.3f}s"
        elif "duration" in t:
            duration = f"{t['duration']:.3f}s"

        # Error detail
        longrepr = ""
        for phase in ("call", "setup", "teardown"):
            phase_data = t.get(phase) or {}
            if isinstance(phase_data, dict) and "longrepr" in phase_data:
                longrepr = str(phase_data["longrepr"])[:400]
                break

        row_bg = "#1e1e2e" if i % 2 == 0 else "#2a2a3e"
        error_html = ""
        if longrepr:
            short = longrepr[:200].replace("<", "&lt;").replace(">", "&gt;")
            error_html = (
                f'<br><code style="font-size:10px;color:#f38ba8;white-space:pre-wrap;'
                f'background:#181825;display:block;padding:4px 6px;border-radius:4px;'
                f'margin-top:4px;">{short}…</code>'
            )

        rows.append(f"""
        <tr style="background:{row_bg}; border-bottom:1px solid #313244;">
          <td style="padding:8px 10px;color:#a6adc8;text-align:center;font-size:12px;">{i}</td>
          <td style="padding:8px 10px;color:#cba6f7;font-size:12px;">{class_name}</td>
          <td style="padding:8px 10px;color:#cdd6f4;font-size:12px;">
            {test_name}{error_html}
          </td>
          <td style="padding:8px 10px;text-align:center;">{badge}</td>
          <td style="padding:8px 10px;color:#a6adc8;text-align:right;font-size:12px;">{duration}</td>
        </tr>""")

    return f"""
<div style="border-radius:10px;overflow:hidden;border:1px solid #313244;">
<table style="width:100%;border-collapse:collapse;font-family:'Segoe UI',system-ui,sans-serif;">
  <thead>
    <tr style="background:#181825;border-bottom:2px solid #45475a;">
      <th style="padding:10px;color:#a6adc8;font-size:12px;text-transform:uppercase;width:40px;">#</th>
      <th style="padding:10px;color:#a6adc8;font-size:12px;text-transform:uppercase;text-align:left;">Class</th>
      <th style="padding:10px;color:#a6adc8;font-size:12px;text-transform:uppercase;text-align:left;">Test</th>
      <th style="padding:10px;color:#a6adc8;font-size:12px;text-transform:uppercase;width:80px;">Result</th>
      <th style="padding:10px;color:#a6adc8;font-size:12px;text-transform:uppercase;width:80px;">Time</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
</div>"""


def _build_coverage_section() -> str:
    """Try to read coverage data if available."""
    cov_file = ROOT / ".coverage"
    if not cov_file.exists():
        return ""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report", "--format=total"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        total = result.stdout.strip()
        return f"""
<div style="background:#313244;border-radius:8px;padding:12px 16px;margin-top:12px;
            font-family:'Segoe UI',system-ui,sans-serif;color:#cdd6f4;">
  <span style="color:#a6adc8;font-size:12px;text-transform:uppercase;letter-spacing:1px;">
    Code Coverage
  </span><br>
  <span style="font-size:24px;font-weight:700;color:#89dceb;">{total}%</span>
  <span style="color:#a6adc8;font-size:12px;"> total</span>
</div>"""
    except Exception:
        return ""


def _build_ollama_status() -> str:
    """Show which Ollama models are available."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=4) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        model_list = "".join(
            f'<li style="color:#a6e3a1;font-size:12px;margin:2px 0;">✓ {m}</li>'
            for m in models
        )
        status = f'<span style="color:#a6e3a1;">● Online</span>'
        content = f"<ul style='margin:8px 0;padding-left:16px;'>{model_list}</ul>"
    except Exception:
        status = f'<span style="color:#f38ba8;">● Offline</span>'
        content = '<p style="color:#f38ba8;font-size:12px;margin:4px 0;">Run: ollama serve</p>'

    return f"""
<div style="background:#313244;border-radius:8px;padding:12px 16px;margin-bottom:12px;
            font-family:'Segoe UI',system-ui,sans-serif;">
  <div style="color:#a6adc8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
    Ollama Status {status}
  </div>
  {content}
</div>"""


def _save_json_report(report: dict, duration: float) -> str:
    """Save the full report as a JSON file and return the path."""
    out = ROOT / "reports" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report["_meta"] = {
        "generated_at": datetime.now().isoformat(),
        "duration_seconds": round(duration, 3),
        "runner": "gradio_report.py",
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return str(out)


# ---------------------------------------------------------------------------
# Gradio event handler
# ---------------------------------------------------------------------------

def run_tests_and_report(marker_filter: str) -> tuple[str, str, str, str]:
    """
    Called by the Gradio button.
    Returns: (summary_html, table_html, log_text, json_path)
    """
    start = time.time()
    report, raw_log = _run_tests(marker_filter)
    duration = time.time() - start

    summary = report.get("summary", {})
    tests   = report.get("tests", [])

    # Normalise summary keys from pytest-json-report
    if not summary and tests:
        from collections import Counter
        counts = Counter(t.get("outcome", "unknown") for t in tests)
        summary = {
            "total": len(tests),
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "skipped": counts.get("skipped", 0),
            "error": counts.get("error", 0),
        }

    summary_html = _build_summary_html(summary, duration)
    summary_html += _build_ollama_status()
    summary_html += _build_coverage_section()

    table_html = _build_results_table(tests)
    json_path  = _save_json_report(report, duration)

    return summary_html, table_html, raw_log, json_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🔬 Quality Triage — Test Report Dashboard

Runs the comprehensive Ollama backend test suite and displays results in real time.

**Test coverage:**
- OllamaAgent live connectivity & streaming
- All 6 tool functions (list_python_files, read_file, code_intel, ml_smells, python_smells, td_classify)
- Tool schema validation
- execute_tool dispatcher
- Config / factory routing
- Error handling & edge cases
- Full agentic tool-call loop (LLM → tool → response)
"""

CUSTOM_CSS = """
body, .gradio-container { background: #1e1e2e !important; }
.prose h1, .prose h2, .prose h3 { color: #cba6f7 !important; }
.prose p, .prose li { color: #cdd6f4 !important; }
footer { display: none !important; }
"""


_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Quality Triage — Test Report") as demo:

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            marker_input = gr.Textbox(
                label="Filter (pytest -k expression)",
                placeholder='e.g. "Config or Schema" — leave blank to run all',
                value="",
                scale=4,
            )
            run_btn = gr.Button("▶  Run Tests", variant="primary", scale=1)

        with gr.Row():
            summary_box = gr.HTML(
                value='<div style="color:#a6adc8;font-style:italic;padding:16px;">'
                      'Click ▶ Run Tests to start...</div>',
                label="Summary",
            )

        with gr.Tabs():
            with gr.Tab("📋 Test Results"):
                results_html = gr.HTML(label="Per-Test Results")

            with gr.Tab("📄 Raw Log"):
                log_box = gr.Code(
                    label="pytest output",
                    language="shell",
                    lines=35,
                )

            with gr.Tab("💾 JSON Report"):
                json_path_box = gr.Textbox(
                    label="Report saved to",
                    interactive=False,
                )
                gr.Markdown(
                    "The full JSON report is written to `reports/` in the project root "
                    "and timestamped so you can keep a history of runs."
                )

        run_btn.click(
            fn=run_tests_and_report,
            inputs=[marker_input],
            outputs=[summary_box, results_html, log_box, json_path_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quality Triage Gradio test dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    # Install pytest-json-report if missing (needed for structured output)
    try:
        import pytest_jsonreport  # noqa: F401
    except ImportError:
        print("Installing pytest-json-report for structured output…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest-json-report", "-q"],
            check=False,
        )

    ui = build_ui()
    print(f"\n🚀  Quality Triage Test Dashboard starting at http://{args.host}:{args.port}")
    print("    Press Ctrl+C to stop.\n")
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=CUSTOM_CSS,
    )
