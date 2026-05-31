"""
FastAPI + server-rendered HTML UI for the code-review agent.

The UI mirrors the CLI: pick a target path/GitHub URL, provider/model, the
detector families + TD categories to run, toggle fix suggestions, run the
review, and view the rendered report, the selection (with skipped families),
and fix diffs — with a gated "apply fixes" action.

It REUSES the same orchestration as the CLI (``code_review_agent.service``)
and the same safe fix engine (``code_review_agent.fixes``); there is no
analysis logic duplicated here. Dependencies are gated behind the ``web``
extra: install with ``uv sync --extra web``.

Run with::

    uv run code-review-web            # binds to 127.0.0.1:8000
"""

from __future__ import annotations

import html
import json
from typing import Any

_INSTALL_HINT = (
    "The web UI needs FastAPI + Uvicorn. Install them with:\n"
    "  uv sync --extra web   (or  pip install \"code-review-agent[web]\")"
)


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def _page(body: str, title: str = "Code Review Agent") -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 980px; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; }}
  fieldset {{ border: 1px solid #8884; border-radius: 8px; margin: 0.75rem 0; padding: 0.75rem 1rem; }}
  legend {{ font-weight: 600; padding: 0 0.4rem; }}
  label.inline {{ display: inline-block; margin-right: 1rem; white-space: nowrap; }}
  input[type=text], select {{ padding: 0.4rem; border-radius: 6px; border: 1px solid #8886; min-width: 16rem; }}
  button {{ padding: 0.5rem 1rem; border-radius: 6px; border: 0; background: #2563eb; color: white;
            font-size: 1rem; cursor: pointer; }}
  button.danger {{ background: #b91c1c; }}
  pre {{ background: #1113; padding: 0.75rem; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
  pre.diff {{ white-space: pre; }}
  .pill {{ display: inline-block; background: #2563eb22; border: 1px solid #2563eb55;
           border-radius: 999px; padding: 0.1rem 0.6rem; margin: 0.1rem; font-size: 0.85rem; }}
  .pill.skip {{ background: #8882; border-color: #8886; color: #888; }}
  .muted {{ color: #888; }}
</style></head><body>
{body}
</body></html>"""


def _controls_form(cfg, service) -> str:
    from code_review_agent.selection import ALL_FAMILIES

    default_model = cfg.ollama.model
    family_boxes = "".join(
        f'<label class="inline"><input type="checkbox" name="checks" value="{_esc(f)}" checked> {_esc(f)}</label>'
        for f in ALL_FAMILIES
    )
    cat_opts = "".join(
        f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in service.known_td_categories()
    )
    return f"""
<h1>🔍 Code Review Agent</h1>
<p class="muted">Local UI — reuses the same tools, agent, and safe fix engine as the CLI.</p>
<form method="post" action="/review">
  <fieldset><legend>Target</legend>
    <input type="text" name="target" placeholder="/path/to/project or https://github.com/owner/repo" required size="60">
  </fieldset>
  <fieldset><legend>Model</legend>
    <label class="inline">Provider
      <select name="provider">
        <option value="ollama" selected>ollama</option>
        <option value="openai">openai</option>
        <option value="anthropic">anthropic</option>
      </select></label>
    <label class="inline">Model <input type="text" name="model" value="{_esc(default_model)}"></label>
  </fieldset>
  <fieldset><legend>Check families</legend>
    {family_boxes}
  </fieldset>
  <fieldset><legend>Technical-debt categories</legend>
    <select name="td_categories" multiple size="6">{cat_opts}</select>
    <div class="muted">Used only when the <code>td</code> family is selected. Default: general.</div>
  </fieldset>
  <fieldset><legend>Fixes &amp; output</legend>
    <label class="inline"><input type="checkbox" name="suggest_fixes" value="1"> Suggest fixes (preview diffs, no writes)</label>
    <label class="inline">Format
      <select name="fmt"><option value="html" selected>html</option><option value="json">json</option></select></label>
  </fieldset>
  <button type="submit">Run review</button>
</form>
"""


def _selection_html(sel: dict) -> str:
    fams = "".join(f'<span class="pill">{_esc(f)}</span>' for f in sel["families"])
    skipped = "".join(f'<span class="pill skip">{_esc(f)} skipped</span>' for f in sel["skipped"])
    cats = ""
    if "td" in sel["families"]:
        cats = "<div>TD categories: " + "".join(
            f'<span class="pill">{_esc(c)}</span>' for c in sel["td_categories"]
        ) + "</div>"
    return f'<div>{fams}{skipped}</div>{cats}'


def _fixes_html(target: str, fixes: list[dict], outcome: dict | None) -> str:
    if not fixes:
        return '<p class="muted">No machine-applicable fix blocks were produced.</p>'
    parts = [f"<p>{len(fixes)} fix suggestion(s). Diffs below are a preview — nothing was written.</p>"]
    diffs = (outcome or {}).get("diffs", [])
    for i, fx in enumerate(fixes):
        diff_text = diffs[i]["diff"] if i < len(diffs) else ""
        parts.append(
            f'<h4>Fix {i + 1}: {_esc(fx.get("description") or fx.get("file"))}</h4>'
            f'<div class="muted">{_esc(fx.get("file"))} lines {_esc(fx.get("start_line"))}-{_esc(fx.get("end_line"))}</div>'
            f'<pre class="diff">{_esc(diff_text)}</pre>'
        )
    # Gated apply form — re-sends the parsed fixes; the server enforces root + confirm.
    parts.append(
        '<form method="post" action="/fixes/apply" '
        'onsubmit="return confirm(\'Apply these fixes to files on disk?\');">'
        f'<input type="hidden" name="target" value="{_esc(target)}">'
        f'<input type="hidden" name="fixes_json" value=\'{_esc(json.dumps(fixes))}\'>'
        '<input type="hidden" name="confirm" value="1">'
        '<button type="submit" class="danger">Apply fixes (writes files, makes .bak backups)</button>'
        '</form>'
    )
    return "".join(parts)


def create_app(config_path: str | None = None):
    """Create the FastAPI app. Raises ImportError if the web extra is missing."""
    try:
        from fastapi import FastAPI, Form
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    from code_review_agent import service
    from code_review_agent.config import get_config
    from code_review_agent.fixes import FixSuggestion, apply_fixes

    app = FastAPI(title="code-review-agent")

    def _cfg():
        return get_config(config_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page(_controls_form(_cfg(), service))

    @app.get("/healthz", response_class=JSONResponse)
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/review", response_class=HTMLResponse)
    def run_review(
        target: str = Form(...),
        provider: str = Form("ollama"),
        model: str = Form(""),
        checks: list[str] = Form(default=[]),
        td_categories: list[str] = Form(default=[]),
        suggest_fixes: list[str] = Form(default=[]),
        fmt: str = Form("html"),
    ):
        cfg = _cfg()
        want_fixes = bool(suggest_fixes)
        result = service.run_review(
            cfg,
            target=target,
            provider=provider or "ollama",
            model=model or None,
            checks=list(checks),
            td_categories=list(td_categories),
            suggest_fixes=want_fixes,
        )
        sel = result["selection"]

        outcome = None
        if want_fixes and result["fix_objects"]:
            # Dry-run preview only: never writes.
            outcome = apply_fixes(result["fix_objects"], target, dry_run=True, confirm=False)

        if fmt == "json":
            return JSONResponse({
                "target": target,
                "selection": {k: sel[k] for k in ("families", "skipped", "td_categories")},
                "report_markdown": result["text"],
                "fixes": result["fixes"],
                "fix_preview": outcome,
            })

        body = (
            '<p><a href="/">&larr; New review</a></p>'
            f"<h1>Report: {_esc(target)}</h1>"
            "<h3>Selection</h3>" + _selection_html(sel) +
            f"<h3>Report</h3><pre>{_esc(result['text'])}</pre>"
            "<h3>Suggested Fixes</h3>" + _fixes_html(target, result["fixes"], outcome)
        )
        return _page(body, title=f"Report — {target}")

    @app.post("/fixes/apply", response_class=HTMLResponse)
    def apply(
        target: str = Form(...),
        fixes_json: str = Form(...),
        confirm: str = Form(""),
    ):
        try:
            raw = json.loads(fixes_json)
        except json.JSONDecodeError:
            raw = []
        suggestions = [
            FixSuggestion(
                file=d.get("file", ""),
                start_line=int(d.get("start_line", 0) or 0),
                end_line=int(d.get("end_line", 0) or 0),
                original=d.get("original", ""),
                replacement=d.get("replacement", ""),
                description=d.get("description", ""),
                finding_id=d.get("finding_id"),
            )
            for d in raw if isinstance(d, dict)
        ]
        # Apply only with explicit confirmation; the engine also confines writes
        # to the target project root.
        outcome = apply_fixes(suggestions, target, dry_run=False, confirm=bool(confirm))
        counts = outcome.get("counts", {})
        rows = "".join(
            f"<li>{_esc(a.get('file'))} lines {_esc(a.get('lines'))} "
            f"(backup {_esc(a.get('backup'))})</li>" for a in outcome.get("applied", [])
        )
        skip_rows = "".join(
            f"<li class='muted'>{_esc(s.get('file'))}: {_esc(s.get('reason'))}</li>"
            for s in outcome.get("skipped", [])
        )
        body = (
            '<p><a href="/">&larr; New review</a></p>'
            f"<h1>Applied fixes</h1>"
            f"<p>{counts.get('applied', 0)} applied, {counts.get('skipped', 0)} skipped.</p>"
            f"<ul>{rows}</ul>"
            f"<h3>Skipped</h3><ul>{skip_rows}</ul>"
        )
        return _page(body, title="Applied fixes")

    return app


def main() -> None:
    """Console-script entry point: serve the UI on localhost."""
    import os

    try:
        import uvicorn
    except ImportError:
        import sys
        print(_INSTALL_HINT, file=sys.stderr)
        sys.exit(1)

    app = create_app()
    host = os.environ.get("CODE_REVIEW_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CODE_REVIEW_WEB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
