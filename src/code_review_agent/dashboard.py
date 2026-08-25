"""Shared review dashboard for the CLI (Rich) and the Gradio / HTML report."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from code_review_agent.reporter import (
    SEVERITY_EMOJI,
    SEVERITY_ORDER,
    ReportData,
    compute_health_score,
    count_by_severity,
    health_label,
)

if TYPE_CHECKING:
    from rich.console import Console

FINDINGS_HEADERS = ["Severity", "ID", "Location", "Category", "Symbol", "Message"]

_BAND_COLORS = {
    "critical": "#A24B2A",
    "high": "#C9892E",
    "medium": "#8A7348",
    "low": "#4A6B52",
    "info": "#3D4A52",
}

_SCORE_COLORS = {
    "healthy": "#4A6B52",
    "needs attention": "#C9892E",
    "at risk": "#A24B2A",
}


def finding_location(finding) -> str:
    loc = f"{finding.file}:{finding.line}"
    if finding.col:
        loc += f":{finding.col}"
    return loc


def findings_table_rows(data: ReportData, limit: int | None = None) -> list[list[str]]:
    rows: list[list[str]] = []
    findings = data.findings[:limit] if limit else data.findings
    for finding in findings:
        rows.append(
            [
                finding.severity.value.upper(),
                finding.finding_id,
                finding_location(finding),
                finding.category,
                finding.symbol or "",
                (finding.message or "")[:160],
            ]
        )
    return rows


def findings_rows_from_payload(
    payload: dict, limit: int | None = None
) -> list[list[str]]:
    """Rebuild the findings table from a saved report JSON object."""
    raw = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    items = raw[:limit] if limit else raw
    rows: list[list[str]] = []
    for finding in items:
        if not isinstance(finding, dict):
            continue
        loc = f"{finding.get('file') or ''}:{finding.get('line') or ''}"
        col = finding.get("col")
        if col:
            loc += f":{col}"
        sev = finding.get("severity")
        if hasattr(sev, "value"):
            sev = sev.value
        rows.append(
            [
                str(sev or "").upper(),
                str(finding.get("finding_id") or finding.get("id") or ""),
                loc,
                str(finding.get("category") or ""),
                str(finding.get("symbol") or ""),
                str(finding.get("message") or "")[:160],
            ]
        )
    return rows


def print_review_dashboard(
    data: ReportData, console: Console, *, top_n: int = 12
) -> None:
    """TTY summary: health, severity strip, top findings. Not a full markdown dump."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    score = compute_health_score(data.findings)
    label = health_label(score)
    counts = count_by_severity(data.findings)
    score_style = {
        "healthy": "green",
        "needs attention": "yellow",
        "at risk": "red",
    }[label]

    header = Text()
    header.append("Health  ", style="dim")
    header.append(f"{score}", style=f"bold {score_style}")
    header.append("/100  ", style=score_style)
    header.append(f"{label}\n", style=f"italic {score_style}")
    header.append(f"{len(data.findings)} findings", style="bold")
    header.append(f"  ·  {data.files_analyzed} files", style="dim")
    if data.duration_s is not None:
        header.append(f"  ·  {data.duration_s:.1f}s", style="dim")
    header.append(f"\n{data.model}", style="dim")

    console.print()
    console.print(
        Panel(header, title="Quality Triage", border_style=score_style, expand=False)
    )

    sev = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    sev.add_column("Severity")
    sev.add_column("Count", justify="right")
    for name in SEVERITY_ORDER:
        n = counts[name]
        style = "dim" if n == 0 else ""
        sev.add_row(f"{SEVERITY_EMOJI[name]} {name}", str(n), style=style)
    console.print(sev)

    if not data.findings:
        console.print("\n[green]No findings at the configured severity.[/green]")
        return

    table = Table(
        title=f"Top findings (first {min(top_n, len(data.findings))})", show_lines=False
    )
    table.add_column("Sev", style="bold", no_wrap=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Location")
    table.add_column("Category")
    table.add_column("Message", max_width=56)
    for finding in data.findings[:top_n]:
        table.add_row(
            f"{SEVERITY_EMOJI[finding.severity.value]} {finding.severity.value}",
            finding.finding_id,
            finding_location(finding),
            finding.category,
            (finding.message or "")[:80],
        )
    console.print()
    console.print(table)

    if data.narrative:
        console.print()
        console.print(
            Panel(
                data.narrative.strip(),
                title="AI synthesis",
                border_style="dim",
            )
        )


def render_html_report(data: ReportData, *, include_snippets: bool = True) -> str:
    """Self-contained HTML report (inspection-bench palette)."""
    score = compute_health_score(data.findings)
    label = health_label(score)
    counts = count_by_severity(data.findings)
    bands = "".join(
        f'<span class="band {sev}" style="flex:{max(counts[sev], 0)};background:{_BAND_COLORS[sev]}" '
        f'title="{sev}: {counts[sev]}"></span>'
        for sev in SEVERITY_ORDER
    )
    if not data.findings:
        bands = '<span class="band empty" style="flex:1;background:#4A6B52"></span>'

    chips = "".join(
        f'<div class="chip"><span class="n" style="color:{_BAND_COLORS[sev]}">{counts[sev]}</span>'
        f'<span class="k">{html.escape(sev)}</span></div>'
        for sev in SEVERITY_ORDER
    )

    findings_html: list[str] = []
    for finding in data.findings:
        loc = html.escape(finding_location(finding))
        snippet = ""
        if include_snippets and finding.code_snippet:
            snippet = f'<pre class="snip">{html.escape(finding.code_snippet.strip()[:800])}</pre>'
        fix = ""
        if finding.how_to_fix:
            fix = f'<p class="fix"><strong>Fix.</strong> {html.escape(finding.how_to_fix)}</p>'
        findings_html.append(
            f'<article class="finding {html.escape(finding.severity.value)}">'
            f'<header><span class="id">{html.escape(finding.finding_id)}</span>'
            f'<span class="cat">{html.escape(finding.category)}</span>'
            f'<span class="sev">{html.escape(finding.severity.value)}</span></header>'
            f'<div class="loc"><code>{loc}</code></div>'
            f"<p>{html.escape(finding.message or '')}</p>"
            f"{snippet}{fix}"
            f"</article>"
        )

    narrative = ""
    if data.narrative:
        narrative = (
            '<section class="narrative"><h2>AI synthesis</h2>'
            f'<pre class="md">{html.escape(data.narrative.strip())}</pre></section>'
        )

    empty = ""
    if not data.findings:
        empty = (
            '<p class="empty">No findings at the configured severity. Detectors still ran; '
            "residual risk is unscanned non-Python files and behaviour the tools cannot see.</p>"
        )

    duration = f"{data.duration_s:.1f}s" if data.duration_s is not None else "—"
    target = html.escape(data.target)
    model = html.escape(data.model)
    tools = html.escape(", ".join(data.tools_run) or "—")
    accent = _SCORE_COLORS[label]
    findings_block = "\n".join(findings_html) or empty

    try:
        from code_review_agent import __version__
    except Exception:
        __version__ = "0.2.0"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Quality Triage — {target}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --paper: #EDE6D9; --ink: #1F1A14; --copper: #A24B2A; --graphite: #3D4A52;
    --sage: #4A6B52; --amber: #C9892E; --rule: #C4B8A5; --accent: {accent};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--paper); color: var(--ink);
    font: 16px/1.5 "Source Sans 3", system-ui, sans-serif;
  }}
  header.mast {{
    padding: 2.5rem 8vw 1.25rem; border-bottom: 1px solid var(--rule);
  }}
  .eyebrow {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--graphite); margin: 0 0 0.4rem;
  }}
  h1 {{
    font-family: Fraunces, Georgia, serif; font-size: 2rem; font-weight: 600;
    margin: 0 0 0.35rem; letter-spacing: -0.02em;
  }}
  .meta {{ color: var(--graphite); font-size: 0.92rem; }}
  .meta code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.85em; }}
  .flag {{
    display: flex; height: 10px; width: 100%; margin: 1.25rem 0 0;
    overflow: hidden; border: 1px solid var(--ink);
  }}
  .band {{ min-width: 0; }}
  .score-row {{
    display: flex; gap: 2rem; align-items: flex-end; flex-wrap: wrap;
    padding: 1.5rem 8vw 0;
  }}
  .score {{
    font-family: Fraunces, Georgia, serif; font-size: 4.5rem; line-height: 0.9;
    color: var(--accent); font-weight: 600;
  }}
  .score small {{ display: block; font-size: 0.9rem; letter-spacing: 0.12em;
    text-transform: uppercase; font-family: "IBM Plex Mono", monospace; margin-top: 0.4rem; }}
  .chips {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .chip {{ min-width: 4.5rem; }}
  .chip .n {{ font-family: Fraunces, serif; font-size: 1.6rem; display: block; }}
  .chip .k {{ font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
    font-family: "IBM Plex Mono", monospace; color: var(--graphite); }}
  main {{ padding: 1.5rem 8vw 4rem; max-width: 1100px; }}
  h2 {{ font-family: Fraunces, serif; font-size: 1.35rem; margin: 2rem 0 0.8rem; }}
  .finding {{
    border-top: 1px solid var(--rule); padding: 1rem 0 1.1rem;
  }}
  .finding header {{ display: flex; gap: 0.75rem; align-items: baseline; flex-wrap: wrap; }}
  .finding .id {{ font-family: "IBM Plex Mono", monospace; font-size: 0.8rem; color: var(--graphite); }}
  .finding .cat {{ font-weight: 700; }}
  .finding .sev {{
    font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; letter-spacing: 0.14em;
    text-transform: uppercase; margin-left: auto;
  }}
  .finding.critical .sev {{ color: var(--copper); }}
  .finding.high .sev {{ color: var(--amber); }}
  .loc code {{ font-family: "IBM Plex Mono", monospace; font-size: 0.85rem; }}
  pre.snip, pre.md {{
    background: #e4dccf; padding: 0.75rem 1rem; overflow-x: auto;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem; white-space: pre-wrap;
  }}
  .empty {{ color: var(--sage); }}
  footer {{ padding: 0 8vw 2rem; color: var(--graphite); font-size: 0.8rem; }}
</style>
</head>
<body>
  <header class="mast">
    <p class="eyebrow">Quality Triage · inspection slip</p>
    <h1>{target}</h1>
    <p class="meta">{html.escape(data.analyzed_at)} · {data.files_analyzed} files · {duration}
      · <code>{model}</code><br/>Tools: {tools}</p>
    <div class="flag" aria-hidden="true">{bands}</div>
  </header>
  <div class="score-row">
    <div class="score">{score}<small>{html.escape(label)} · {len(data.findings)} findings</small></div>
    <div class="chips">{chips}</div>
  </div>
  <main>
    {narrative}
    <h2>Findings</h2>
    {findings_block}
  </main>
  <footer>Generated by code-review-agent v{__version__}</footer>
</body>
</html>
"""
