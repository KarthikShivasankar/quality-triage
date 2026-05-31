"""
CLI entry point for code-review-agent.

Commands:
  review PATH|URL          Full AI review (auto-detects GitHub URLs)
  ask QUESTION             Ask the agent a question
  analyze-file FILE        Deep-dive on a single file
  run-tool TOOL [opts]     On-demand tool execution
  interactive PATH         Interactive tool selector
  show-config              Print resolved configuration
  list-tools               List all available tools
  ollama-models            List models in local Ollama instance
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from code_review_agent.selection import ALL_FAMILIES


def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 tolerant.

    LLM output frequently contains emoji/unicode; on a legacy Windows console
    (cp1252) rich would otherwise raise UnicodeEncodeError mid-stream. Reconfigure
    to UTF-8 with ``errors='replace'`` so streaming never crashes the review.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_streams()

console = Console()

PROVIDER_CHOICE = click.Choice(["ollama", "openai", "anthropic"])
FAMILY_CHOICE = click.Choice(ALL_FAMILIES)
SEVERITY_CHOICE = click.Choice(["critical", "high", "medium", "low", "info"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str | None):
    from code_review_agent.config import get_config, reset_config
    reset_config()
    return get_config(config_path)


def _model_for_provider(cfg, provider, model_override):
    if model_override:
        return model_override
    if provider == "openai":
        return cfg.openai.model
    if provider == "anthropic":
        return cfg.anthropic.model
    return cfg.ollama.model


def _make_agent(cfg, provider_override=None, model_override=None, base_url=None, api_key=None, families=None):
    from code_review_agent.config import resolve_api_key
    from code_review_agent.service import build_agent

    provider = provider_override or cfg.provider

    # Friendly pre-flight key checks (the agent also raises, but we want a clean message).
    if provider == "anthropic":
        key = api_key or resolve_api_key(cfg.anthropic.api_key, cfg.anthropic.api_key_env, "ANTHROPIC_API_KEY")
        if not key:
            console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set.")
            console.print("  Use Ollama instead: [cyan]--provider ollama[/cyan]")
            sys.exit(1)
    elif provider == "openai":
        key = api_key or resolve_api_key(cfg.openai.api_key, cfg.openai.api_key_env, "OPENAI_API_KEY")
        if not key:
            console.print(
                f"[red]Error:[/red] no API key for the 'openai' provider "
                f"(env [cyan]{cfg.openai.api_key_env}[/cyan] or OPENAI_API_KEY)."
            )
            console.print("  Pass [cyan]--api-key[/cyan], or use [cyan]--provider ollama[/cyan].")
            sys.exit(1)

    model = _model_for_provider(cfg, provider, model_override)
    resolved_base = base_url or (
        cfg.openai.base_url if provider == "openai"
        else cfg.ollama.base_url if provider == "ollama" else None
    )
    sel_note = f"  Checks: [bold]{','.join(families)}[/bold]" if families else ""
    console.print(
        f"[dim]Provider: [bold]{provider}[/bold]  Model: [bold]{model}[/bold]"
        + (f"  Base URL: [bold]{resolved_base}[/bold]" if resolved_base else "")
        + sel_note
        + "[/dim]"
    )
    return build_agent(
        cfg, provider, model_override, families,
        base_url=base_url, api_key=api_key,
    )


def _stream(agent, gen_fn, *args, output_path=None, **kwargs):
    """Stream agent response, optionally saving to file."""
    collected: list[str] = []
    try:
        for chunk in gen_fn(*args, **kwargs):
            console.print(chunk, end="", markup=False)
            collected.append(chunk)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted.[/yellow]")

    text = "".join(collected)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        console.print(f"\n\n[green]Saved to:[/green] {output_path}")
    return text


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="code-review-agent")
@click.option(
    "--config", "-C",
    default=None,
    metavar="PATH",
    help="Path to config.yaml (default: ./config.yaml)",
    is_eager=True,
    expose_value=True,
    envvar="CODE_REVIEW_CONFIG",
)
@click.pass_context
def main(ctx, config):
    """AI-powered code review agent.

    \b
    Providers:
      ollama     local Ollama, no API key (default)
      openai     any OpenAI-compatible API: OpenAI, Groq, OpenRouter,
                 Together, Fireworks, Mistral, llama.cpp, vLLM, LM Studio
      anthropic  Anthropic Claude
    \b
    Configure everything in config.yaml — run `show-config` to inspect,
    `providers` to see backends, and `doctor` for a health check.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

@main.command()
@click.argument("target")   # local path OR GitHub URL
@click.option("--context", "-c", default="", help="Extra context / focus areas")
@click.option("--output", "-o", default=None, help="Save report to file")
@click.option("--provider", "-p", type=PROVIDER_CHOICE, default=None)
@click.option("--model", "-m", default=None, help="Override model name")
@click.option("--base-url", default=None, help="Override base URL (openai/ollama providers)")
@click.option("--api-key", default=None, help="Override API key (openai/anthropic providers)")
@click.option("--check", "-k", "checks", multiple=True, type=FAMILY_CHOICE,
              help="Detector families to run (repeatable). Default: config / all.")
@click.option("--td-category", "td_categories", multiple=True,
              help="TD categories to classify (repeatable). Default: config / general.")
@click.option("--min-severity", type=SEVERITY_CHOICE, default=None,
              help="Ask the report to focus on findings at or above this severity.")
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default=None,
              help="Saved output format (default: markdown).")
@click.option("--suggest-fixes", is_flag=True, default=False,
              help="Ask the agent for machine-applicable fix blocks (suggest-only).")
@click.option("--fix-dry-run", is_flag=True, default=False,
              help="Show fix diffs without writing (implies --suggest-fixes).")
@click.option("--apply-fixes", is_flag=True, default=False,
              help="Apply parsed fixes (requires confirmation; implies --suggest-fixes).")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the apply confirmation prompt.")
@click.option("--keep-clone", is_flag=True, default=False, help="Keep cloned GitHub repo")
@click.pass_context
def review(ctx, target, context, output, provider, model, base_url, api_key,
           checks, td_categories, min_severity, fmt, suggest_fixes, fix_dry_run,
           apply_fixes, yes, keep_clone):
    """Full AI code review on a local PATH or GitHub URL.

    \b
    Examples:
      code-review review ./my_project
      code-review review . --check ml --check structural
      code-review review . --check td --td-category security --td-category design
      code-review review . --suggest-fixes --fix-dry-run
      code-review review . --apply-fixes --yes
      code-review review https://github.com/owner/repo --provider anthropic
    """
    cfg = _load_cfg(ctx.obj.get("config_path"))
    from code_review_agent import service

    # Handle GitHub URLs
    from code_review_agent.github_utils import is_github_url, clone_repo, cleanup_repo
    cloned = None
    if is_github_url(target):
        console.print(f"[cyan]Cloning:[/cyan] {target}")
        try:
            cloned = clone_repo(
                target,
                depth=cfg.github.depth,
                timeout=cfg.github.timeout,
            )
            console.print(f"[green]Cloned to:[/green] {cloned.local_path}  (commit {cloned.commit_sha[:8]})")
            review_path = cloned.local_path
        except Exception as e:
            console.print(f"[red]Clone failed:[/red] {e}")
            sys.exit(1)
    else:
        if not Path(target).exists():
            console.print(f"[red]Error:[/red] Path does not exist: {target}")
            sys.exit(1)
        review_path = str(Path(target).resolve())

    # Resolve the analysis selection and build the prompt context.
    sel = service.resolve_selection(cfg, list(checks), list(td_categories))
    if sel["unknown_td_categories"]:
        console.print(f"[yellow]Ignoring unknown TD categories:[/yellow] {sel['unknown_td_categories']}")
    fmt = fmt or cfg.review.output_format
    want_fixes = suggest_fixes or fix_dry_run or apply_fixes or cfg.review.suggest_fixes

    user_ctx = context
    if min_severity:
        user_ctx = (user_ctx + f"\nFocus on findings at severity {min_severity.upper()} or higher.").strip()
    extra_context = service.review_context(user_ctx, sel, want_fixes)

    agent = _make_agent(cfg, provider, model, base_url=base_url, api_key=api_key, families=sel["families"])

    console.print(Panel(
        f"[bold cyan]Code Review[/bold cyan]\n"
        f"Target: [green]{target}[/green]\n"
        f"Path:   [dim]{review_path}[/dim]\n"
        f"Checks: [yellow]{', '.join(sel['families'])}[/yellow]"
        + (f"   Skipped: [dim]{', '.join(sel['skipped'])}[/dim]" if sel['skipped'] else "")
        + (f"\nTD categories: [yellow]{', '.join(sel['td_categories'])}[/yellow]" if 'td' in sel['families'] else ""),
        expand=False,
    ))
    console.print()

    start = time.time()
    text = ""
    try:
        # Markdown: stream to the file directly. JSON: capture, then write wrapped.
        stream_output = output if (fmt == "markdown") else None
        text = _stream(agent, agent.review, review_path, extra_context=extra_context, output_path=stream_output)
    finally:
        if cloned and not keep_clone:
            cleanup_repo(cloned)
            console.print(f"\n[dim]Cleaned up clone: {cloned.local_path}[/dim]")

    fixes_payload = None
    if want_fixes:
        mode = "apply" if apply_fixes else "dry-run"
        fixes_payload = _handle_review_fixes(text, review_path, mode=mode, assume_yes=yes)

    if fmt == "json" and output:
        _save_review_json(output, target, sel, text, fixes_payload)

    console.print(f"\n[dim]Finished in {time.time()-start:.1f}s[/dim]")


def _handle_review_fixes(text: str, project_root: str, mode: str, assume_yes: bool):
    """Parse fix blocks from the review text and preview/apply them safely."""
    from code_review_agent.fixes import apply_fixes as _apply, parse_fix_blocks, render_fixes_markdown

    suggestions = parse_fix_blocks(text)
    if not suggestions:
        console.print("\n[dim]No machine-applicable fix blocks were produced.[/dim]")
        return {"fixes": [], "outcome": None}

    console.print(f"\n[bold cyan]Parsed {len(suggestions)} fix suggestion(s).[/bold cyan]")

    if mode == "apply":
        confirm = assume_yes or click.confirm(
            f"Apply {len(suggestions)} fix(es) to files under {project_root}?", default=False
        )
        outcome = _apply(suggestions, project_root, dry_run=not confirm, confirm=confirm)
    else:
        # Dry-run preview: never writes.
        outcome = _apply(suggestions, project_root, dry_run=True, confirm=False)

    for d in outcome.get("diffs", []):
        console.print(f"\n[yellow]{d['file']}[/yellow] (lines {d['lines']}) — {d.get('description', '')}")
        if d.get("diff"):
            syntax = Syntax(d["diff"], "diff", theme="monokai")
            console.print(syntax)
    counts = outcome.get("counts", {})
    console.print(
        f"\n[green]Fixes:[/green] {counts.get('applied', 0)} applied, "
        f"{counts.get('skipped', 0)} skipped, {counts.get('diffs', 0)} diff(s)."
    )
    if outcome.get("applied"):
        console.print("[dim]Backups written next to each modified file (.bak).[/dim]")
    return {
        "fixes": [s.to_dict() for s in suggestions],
        "outcome": outcome,
        "fixes_markdown": render_fixes_markdown(suggestions, outcome),
    }


def _save_review_json(output, target, sel, text, fixes_payload):
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "selection": {
            "families": sel["families"],
            "skipped": sel["skipped"],
            "td_categories": sel["td_categories"],
        },
        "report_markdown": text,
        "fixes": (fixes_payload or {}).get("fixes", []),
        "fix_outcome": (fixes_payload or {}).get("outcome"),
    }
    out_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    console.print(f"\n[green]Saved JSON to:[/green] {output}")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@main.command()
@click.argument("question")
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", type=PROVIDER_CHOICE, default=None)
@click.option("--model", "-m", default=None)
@click.option("--base-url", default=None, help="Override base URL (openai/ollama providers)")
@click.option("--api-key", default=None, help="Override API key (openai/anthropic providers)")
@click.pass_context
def ask(ctx, question, output, provider, model, base_url, api_key):
    """Ask the agent a code quality question."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    agent = _make_agent(cfg, provider, model, base_url=base_url, api_key=api_key)
    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    _stream(agent, agent.ask, question, output_path=output)


# ---------------------------------------------------------------------------
# analyze-file
# ---------------------------------------------------------------------------

@main.command("analyze-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", type=PROVIDER_CHOICE, default=None)
@click.option("--model", "-m", default=None)
@click.option("--base-url", default=None, help="Override base URL (openai/ollama providers)")
@click.option("--api-key", default=None, help="Override API key (openai/anthropic providers)")
@click.option("--check", "-k", "checks", multiple=True, type=FAMILY_CHOICE,
              help="Detector families to run (repeatable). Default: all.")
@click.option("--td-category", "td_categories", multiple=True,
              help="TD categories to classify (repeatable).")
@click.option("--suggest-fixes", is_flag=True, default=False, help="Ask for machine-applicable fixes.")
@click.option("--fix-dry-run", is_flag=True, default=False, help="Show fix diffs without writing.")
@click.option("--apply-fixes", is_flag=True, default=False, help="Apply fixes (requires confirmation).")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the apply confirmation prompt.")
@click.pass_context
def analyze_file(ctx, file_path, output, provider, model, base_url, api_key,
                 checks, td_categories, suggest_fixes, fix_dry_run, apply_fixes, yes):
    """Deep-dive review of a single Python file."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    from code_review_agent import service

    sel = service.resolve_selection(cfg, list(checks), list(td_categories))
    want_fixes = suggest_fixes or fix_dry_run or apply_fixes
    agent = _make_agent(cfg, provider, model, base_url=base_url, api_key=api_key, families=sel["families"])
    abs_path = str(Path(file_path).resolve())
    prompt = (
        f"Perform a detailed code review of `{abs_path}` with exact line:col references."
        "\n\n" + service.review_context("", sel, want_fixes)
    )
    console.print(Panel(
        f"[bold cyan]File Review[/bold cyan]\n[green]{abs_path}[/green]\n"
        f"Checks: [yellow]{', '.join(sel['families'])}[/yellow]",
        expand=False,
    ))
    console.print()
    text = _stream(agent, agent.ask, prompt, output_path=output if not want_fixes else None)
    if want_fixes:
        mode = "apply" if apply_fixes else "dry-run"
        _handle_review_fixes(text, str(Path(file_path).resolve().parent), mode=mode, assume_yes=yes)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(text, encoding="utf-8")
            console.print(f"\n[green]Saved to:[/green] {output}")


# ---------------------------------------------------------------------------
# run-tool  (on-demand tool execution)
# ---------------------------------------------------------------------------

@main.group("run-tool")
def run_tool():
    """On-demand execution of individual analysis tools."""


@run_tool.command("ml-smells")
@click.argument("path", type=click.Path(exists=True))
@click.option("--ignore", "-i", multiple=True, help="Dirs to ignore")
@click.option("--min-severity", type=SEVERITY_CHOICE, default=None, help="Only show findings at/above this severity.")
@click.option("--output", "-o", default=None, help="Save JSON output to file")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_ml_smells(ctx, path, ignore, min_severity, output, fmt):
    """Detect ML-specific anti-patterns (data leakage, magic numbers, etc.)."""
    from code_review_agent.tools import detect_ml_smells
    _load_cfg(ctx.obj.get("config_path"))

    with console.status("[cyan]Running ML smell detector…[/cyan]"):
        result = detect_ml_smells(str(Path(path).resolve()), ignore_dirs=list(ignore) or None)

    if min_severity:
        result = _filter_result_by_severity(result, min_severity)
    _print_tool_result(result, fmt, output, "ML Smells")


@run_tool.command("python-smells")
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", "analysis_type",
              type=click.Choice(["code", "architectural", "structural", "all"]),
              default="all", show_default=True)
@click.option("--ignore", "-i", multiple=True)
@click.option("--min-severity", type=SEVERITY_CHOICE, default=None, help="Only show findings at/above this severity.")
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_python_smells(ctx, path, analysis_type, ignore, min_severity, output, fmt):
    """Detect code/architectural/structural Python code smells."""
    from code_review_agent.tools import detect_python_smells
    _load_cfg(ctx.obj.get("config_path"))

    with console.status(f"[cyan]Running Python smell detector ({analysis_type})…[/cyan]"):
        result = detect_python_smells(
            str(Path(path).resolve()),
            analysis_type=analysis_type,
            ignore_dirs=list(ignore) or None,
        )

    if min_severity:
        result = _filter_result_by_severity(result, min_severity)
    _print_tool_result(result, fmt, output, "Python Smells")


@run_tool.command("classify-td")
@click.option("--text", "-t", multiple=True, help="Text snippet to classify (repeatable)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="File with one snippet per line")
@click.option("--category", default=None,
              help="Debt category: general (default), code, design, test, security, documentation, …")
@click.option("--model-path", default=None, help="HuggingFace model ID override")
@click.option("--onnx-path", default=None, help="Local .onnx model path (skip download)")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_classify_td(ctx, text, from_file, category, model_path, onnx_path, device, output, fmt):
    """Classify text snippets for technical debt (binary, per-category model).

    \b
    Examples:
      code-review run-tool classify-td --text "TODO: refactor this hack"
      code-review run-tool classify-td --category security --from-file notes.txt
    """
    from code_review_agent.tools import classify_technical_debt
    _load_cfg(ctx.obj.get("config_path"))

    texts = list(text)
    if from_file:
        texts += [
            ln.strip()
            for ln in Path(from_file).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    if not texts:
        console.print("[red]Error:[/red] Provide --text or --from-file")
        sys.exit(1)

    console.print(
        "[dim]Note: first run downloads the model (needs network); this can take a while.[/dim]"
    )
    with console.status("[cyan]Classifying technical debt…[/cyan]"):
        result = classify_technical_debt(
            texts, category=category, model_path=model_path,
            onnx_path=onnx_path, device=device,
        )

    _print_tool_result(result, fmt, output, "Technical Debt Classification")


@run_tool.command("classify-td-all")
@click.option("--text", "-t", multiple=True, help="Text snippet to classify (repeatable)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="File with one snippet per line")
@click.option("--category", "categories", multiple=True,
              help="Restrict to these categories (repeatable); default: all 21")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default=None)
@click.option("--output", "-o", default=None)
@click.pass_context
def tool_classify_td_all(ctx, text, from_file, categories, device, output):
    """Classify snippets against EVERY TD category model (multi-label sweep).

    \b
    Downloads each category model on first use. Returns, per snippet, the list
    of debt categories whose binary model fired (class==1).
    """
    from code_review_agent.tools import classify_technical_debt_all
    _load_cfg(ctx.obj.get("config_path"))

    texts = list(text)
    if from_file:
        texts += [
            ln.strip()
            for ln in Path(from_file).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    if not texts:
        console.print("[red]Error:[/red] Provide --text or --from-file")
        sys.exit(1)

    console.print("[dim]Note: downloads one model per category on first run.[/dim]")
    with console.status("[cyan]Sweeping all TD categories…[/cyan]"):
        result = classify_technical_debt_all(texts, categories=list(categories) or None, device=device)
    _print_tool_result(result, "json", output, "TD Multi-Category Sweep")


@run_tool.command("classify-td-ensemble")
@click.option("--text", "-t", multiple=True, help="Text snippet to classify (repeatable)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="File with one snippet per line")
@click.option("--category", "categories", multiple=True, help="Category model to add to the ensemble (repeatable)")
@click.option("--model", "model_names", multiple=True, help="HuggingFace model id to add (repeatable)")
@click.option("--weight", "weights", multiple=True, type=float, help="Per-model weight (repeatable, matches model order)")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default=None)
@click.option("--backend", type=click.Choice(["auto", "onnx", "torch"]), default=None,
              help="Ensemble backend: onnx (torch-free, default), torch, or auto")
@click.option("--output", "-o", default=None)
@click.pass_context
def tool_classify_td_ensemble(ctx, text, from_file, categories, model_names, weights, device, backend, output):
    """Classify snippets with a WEIGHTED ENSEMBLE of TD models.

    Runs on the native torch-free ONNX ensemble by default; pass
    --backend torch to force the PyTorch engine.

    \b
    Example:
      code-review run-tool classify-td-ensemble -t "TODO: hack" \\
        --category security --category design --weight 0.6 --weight 0.4
    """
    from code_review_agent.tools import classify_technical_debt_ensemble
    _load_cfg(ctx.obj.get("config_path"))

    texts = list(text)
    if from_file:
        texts += [
            ln.strip()
            for ln in Path(from_file).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    if not texts:
        console.print("[red]Error:[/red] Provide --text or --from-file")
        sys.exit(1)

    console.print("[dim]Note: downloads each ensemble model on first run.[/dim]")
    with console.status("[cyan]Running TD ensemble…[/cyan]"):
        result = classify_technical_debt_ensemble(
            texts,
            model_names=list(model_names) or None,
            categories=list(categories) or None,
            weights=list(weights) or None,
            device=device,
            backend=backend,
        )
    _print_tool_result(result, "json", output, "TD Ensemble Classification")


@run_tool.command("td-categories")
@click.pass_context
def tool_td_categories(ctx):
    """List the available technical-debt categories and their HF model ids."""
    from code_review_agent.tools import list_td_categories
    _load_cfg(ctx.obj.get("config_path"))

    result = list_td_categories()
    table = Table(title="Technical-Debt Categories", show_lines=False)
    table.add_column("Category", style="green")
    table.add_column("Label", style="yellow")
    table.add_column("HF Model", style="cyan")
    for c in result["categories"]:
        table.add_row(c["category"], c["label"], c["model"])
    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(result['categories'])} categories, "
                  f"{len(result['aliases'])} aliases[/dim]\n")


@run_tool.command("td-issues")
@click.argument("repo")
@click.option("--category", default=None, help="Debt category (default: general)")
@click.option("--state", type=click.Choice(["open", "closed", "all"]), default="all", show_default=True)
@click.option("--limit", default=50, show_default=True, type=int)
@click.option("--all", "fetch_all", is_flag=True, help="Fetch every issue (paginates)")
@click.option("--token", default=None, help="GitHub token (or set GITHUB_TOKEN)")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="json")
@click.pass_context
def tool_td_issues(ctx, repo, category, state, limit, fetch_all, token, device, output, fmt):
    """Fetch GitHub issues from owner/REPO and classify them for technical debt.

    \b
    Example:
      code-review run-tool td-issues pandas-dev/pandas --category defect --limit 30
    """
    from code_review_agent.tools import classify_github_issues
    _load_cfg(ctx.obj.get("config_path"))

    console.print("[dim]Fetching issues (network) and classifying (downloads model on first run)…[/dim]")
    with console.status(f"[cyan]Classifying issues from {repo}…[/cyan]"):
        result = classify_github_issues(
            repo, category=category, state=state, limit=limit,
            fetch_all=fetch_all, token=token, device=device,
        )
    _print_tool_result(result, fmt, output, "GitHub Issues — Technical Debt")


@run_tool.command("td-split")
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, help="Directory to write train/test CSVs")
@click.option("--test-size", default=0.2, show_default=True, type=float)
@click.option("--random-state", default=42, show_default=True, type=int)
@click.option("--repo-column", default=None, help="Column with repository info (enables top-repo extraction)")
@click.option("--hf-dataset", "is_hf", is_flag=True, help="data_file is a HuggingFace dataset name")
@click.option("--numeric-labels", is_flag=True, help="Labels are already 0/1")
@click.pass_context
def tool_td_split(ctx, data_file, output_dir, test_size, random_state, repo_column, is_hf, numeric_labels):
    """Split/save a dataset for TD training (tdsuite split-data)."""
    from code_review_agent.tools import td_split_data
    _load_cfg(ctx.obj.get("config_path"))

    with console.status("[cyan]Splitting data…[/cyan]"):
        result = td_split_data(
            data_file, output_dir, test_size=test_size, random_state=random_state,
            repo_column=repo_column, is_huggingface_dataset=is_hf, is_numeric_labels=numeric_labels,
        )
    _print_tool_result(result, "json", None, "TD Data Split")


@run_tool.command("td-export-onnx")
@click.option("--model-name", default=None, help="HuggingFace model id to export")
@click.option("--model-path", default=None, help="Local model checkpoint dir to export")
@click.option("--output", "-o", required=True, help="Destination .onnx file path")
@click.option("--max-length", default=512, show_default=True, type=int)
@click.option("--opset", default=14, show_default=True, type=int)
@click.pass_context
def tool_td_export_onnx(ctx, model_name, model_path, output, max_length, opset):
    """Export a transformer TD model to ONNX for CPU inference (tdsuite export-onnx)."""
    from code_review_agent.tools import td_export_onnx
    _load_cfg(ctx.obj.get("config_path"))

    console.print("[dim]Note: downloads the source model and requires the 'onnx' package.[/dim]")
    with console.status("[cyan]Exporting to ONNX…[/cyan]"):
        result = td_export_onnx(
            output, model_name=model_name, model_path=model_path,
            max_length=max_length, opset=opset,
        )
    _print_tool_result(result, "json", None, "TD ONNX Export")


@run_tool.command("td-train")
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--model-name", required=True, help="Base HF model to fine-tune (e.g. roberta-base)")
@click.option("--output-dir", "-o", required=True, help="Directory to save the trained model")
@click.option("--epochs", "num_epochs", default=3, show_default=True, type=int)
@click.option("--batch-size", default=16, show_default=True, type=int)
@click.option("--learning-rate", default=2e-5, show_default=True, type=float)
@click.option("--positive-category", default=None, help="Positive category for binary labels")
@click.option("--numeric-labels", is_flag=True, help="Labels are already 0/1")
@click.option("--hf-dataset", "is_hf", is_flag=True, help="data_file is a HuggingFace dataset name")
@click.option("--cross-validation", is_flag=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default=None)
@click.pass_context
def tool_td_train(ctx, data_file, model_name, output_dir, num_epochs, batch_size,
                  learning_rate, positive_category, numeric_labels, is_hf, cross_validation, device):
    """Train a binary TD classifier (tdsuite train). Requires torch; GPU strongly recommended."""
    from code_review_agent.tools import td_train
    _load_cfg(ctx.obj.get("config_path"))

    console.print("[yellow]Training requires torch and is very slow on CPU.[/yellow]")
    result = td_train(
        data_file, model_name, output_dir,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        positive_category=positive_category, numeric_labels=numeric_labels,
        is_huggingface_dataset=is_hf, cross_validation=cross_validation, device=device,
    )
    _print_tool_result(result, "json", None, "TD Training")


@run_tool.command("code-intel")
@click.argument("path", type=click.Path(exists=True))
@click.option("--symbol", "-s", default=None, help="Look up this symbol")
@click.option("--usages", "-u", default=None, help="Find all usages of this symbol")
@click.option("--metrics", "metrics_only", is_flag=True, help="Show function metrics only")
@click.option("--imports", "import_graph", is_flag=True, help="Show import graph")
@click.option("--top-n", default=15, show_default=True)
@click.option("--ignore", "-i", multiple=True)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_code_intel(ctx, path, symbol, usages, metrics_only, import_graph, top_n, ignore, output, fmt):
    """AST code intelligence: symbols, metrics, imports, usages."""
    from code_review_agent.tools import analyze_code_intelligence
    _load_cfg(ctx.obj.get("config_path"))

    with console.status("[cyan]Analyzing code intelligence…[/cyan]"):
        result = analyze_code_intelligence(
            str(Path(path).resolve()),
            symbol=symbol,
            find_usages_of=usages,
            metrics_only=metrics_only,
            import_graph=import_graph,
            ignore_dirs=list(ignore) or None,
            top_n=top_n,
        )

    if fmt == "table":
        _print_code_intel_table(result, path)
    else:
        _print_tool_result(result, fmt, output, "Code Intelligence")


@run_tool.command("list-files")
@click.argument("path", type=click.Path(exists=True))
@click.option("--ignore", "-i", multiple=True)
@click.pass_context
def tool_list_files(ctx, path, ignore):
    """List all Python files in a project directory."""
    from code_review_agent.tools import list_python_files
    _load_cfg(ctx.obj.get("config_path"))

    result = list_python_files(str(Path(path).resolve()), ignore_dirs=list(ignore) or None)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    table = Table(title=f"Python files in {path}", show_lines=False)
    table.add_column("File", style="green")
    table.add_column("Size", justify="right", style="dim")
    for f in result.get("files", []):
        table.add_row(f["path"], f"{f['size_kb']} KB")
    console.print(table)
    console.print(f"\n[dim]Total: {result['total_files']} files[/dim]")


@run_tool.command("read-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--max-lines", default=None, type=int)
@click.pass_context
def tool_read_file(ctx, file_path, max_lines):
    """Read a Python file with line numbers."""
    from code_review_agent.tools import read_file
    _load_cfg(ctx.obj.get("config_path"))

    result = read_file(str(Path(file_path).resolve()), max_lines=max_lines)
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(f"[dim]{result['file']} — {result['shown_lines']}/{result['total_lines']} lines[/dim]\n")
    syntax = Syntax(result["content"], "python", line_numbers=False, theme="monokai")
    console.print(syntax)
    if result.get("truncated"):
        console.print(f"\n[yellow]… truncated at {result['shown_lines']} lines[/yellow]")


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------

@main.command()
@click.argument("target")
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", type=PROVIDER_CHOICE, default=None)
@click.option("--model", "-m", default=None)
@click.option("--base-url", default=None, help="Override base URL (openai/ollama providers)")
@click.option("--api-key", default=None, help="Override API key (openai/anthropic providers)")
@click.pass_context
def interactive(ctx, target, output, provider, model, base_url, api_key):
    """Interactive tool selector — choose which tools to run, then get AI synthesis."""
    cfg = _load_cfg(ctx.obj.get("config_path"))

    from code_review_agent.github_utils import is_github_url, clone_repo
    cloned = None
    if is_github_url(target):
        console.print(f"[cyan]Cloning:[/cyan] {target}")
        cloned = clone_repo(target, depth=cfg.github.depth, timeout=cfg.github.timeout)
        review_path = cloned.local_path
    else:
        review_path = str(Path(target).resolve())

    console.print(Panel(f"[bold cyan]Interactive Code Review[/bold cyan]\nTarget: [green]{target}[/green]", expand=False))
    console.print()

    tool_choices = {
        "1": ("List Python files",         "list_python_files",         {"directory": review_path}),
        "2": ("Code Intelligence (AST)",   "analyze_code_intelligence", {"path": review_path}),
        "3": ("Python smells (all)",       "detect_python_smells",      {"path": review_path}),
        "4": ("ML smells",                 "detect_ml_smells",          {"path": review_path}),
        "5": ("Classify technical debt",   None,                        None),  # special
    }

    console.print("[bold]Available tools:[/bold]")
    for key, (name, _, _) in tool_choices.items():
        console.print(f"  [{key}] {name}")
    console.print("  [a] Run ALL tools")
    console.print("  [q] Quit")
    console.print()

    selected = click.prompt("Select tools (comma-separated, e.g. 1,3,4 or a)", default="a")

    from code_review_agent.tools import (
        classify_technical_debt, execute_tool,
    )

    keys_to_run = list(tool_choices.keys()) if selected.strip().lower() == "a" \
                  else [k.strip() for k in selected.split(",") if k.strip() in tool_choices]

    results: dict[str, Any] = {}
    td_texts: list[str] = []

    for key in keys_to_run:
        name, fn_name, kwargs = tool_choices[key]
        if fn_name is None:
            continue  # TD handled after
        with console.status(f"[cyan]Running: {name}…[/cyan]"):
            result = execute_tool(fn_name, kwargs or {})
        results[fn_name] = json.loads(result)
        console.print(f"  [green]✓[/green] {name}")

    # TD: extract texts from read files if available
    if "5" in keys_to_run:
        py_files_result = results.get("list_python_files", {})
        py_files = py_files_result.get("files", [])[:5]  # top 5
        for pf in py_files:
            from code_review_agent.tools import read_file as _rf
            fc = _rf(pf["abs_path"], max_lines=200)
            content = fc.get("content", "")
            for line in content.splitlines():
                stripped = line.split("|", 1)[-1].strip()
                if any(marker in stripped.upper() for marker in ("TODO", "FIXME", "HACK", "NOTE", "XXX")):
                    td_texts.append(stripped[:200])
        if td_texts:
            with console.status("[cyan]Classifying technical debt…[/cyan]"):
                td_result_raw = classify_technical_debt(td_texts)
            results["classify_technical_debt"] = td_result_raw
            console.print(f"  [green]✓[/green] Technical debt ({len(td_texts)} snippets)")

    console.print()

    # AI synthesis?
    if click.confirm("Run AI synthesis of results?", default=True):
        agent = _make_agent(cfg, provider, model, base_url=base_url, api_key=api_key)
        summary = json.dumps(
            {k: v for k, v in results.items()},
            default=str, indent=2
        )[:8000]  # truncate to avoid context overflow
        prompt = (
            f"I have run code analysis tools on the project at `{review_path}` and collected these results:\n\n"
            f"```json\n{summary}\n```\n\n"
            "Please synthesise these findings into a structured code review report with:\n"
            "1. Executive summary\n2. Critical issues with exact file:line locations\n"
            "3. Prioritised recommendations\n4. Improvement roadmap"
        )
        console.print()
        _stream(agent, agent.ask, prompt, output_path=output)

    if cloned:
        if click.confirm(f"\nDelete clone at {cloned.local_path}?", default=True):
            from code_review_agent.github_utils import cleanup_repo as cr
            cr(cloned)


# ---------------------------------------------------------------------------
# show-config
# ---------------------------------------------------------------------------

@main.command("show-config")
@click.pass_context
def show_config(ctx):
    """Print the resolved configuration."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()
                    if not k.startswith("_")}
        return obj

    d = _to_dict(cfg)
    d.pop("_raw", None)
    d.pop("_source", None)

    import yaml
    console.print(f"\n[dim]Config source: [bold]{cfg._source}[/bold][/dim]\n")
    syntax = Syntax(yaml.dump(d, default_flow_style=False), "yaml", theme="monokai")
    console.print(syntax)


# ---------------------------------------------------------------------------
# list-tools
# ---------------------------------------------------------------------------

@main.command("list-tools")
def list_tools():
    """List all available analysis tools."""
    from code_review_agent.tools import TOOL_REGISTRY, TOOL_DEFINITIONS_OPENAI

    table = Table(title="Available Tools", show_lines=True)
    table.add_column("Tool", style="green", no_wrap=True)
    table.add_column("run-tool command", style="cyan", no_wrap=True)
    table.add_column("Description")

    cmd_map = {
        "detect_ml_smells": "ml-smells PATH",
        "detect_python_smells": "python-smells PATH [--type all]",
        "classify_technical_debt": "classify-td --text TEXT [--category C]",
        "classify_technical_debt_all": "classify-td-all --text TEXT",
        "classify_technical_debt_ensemble": "classify-td-ensemble --category A --category B",
        "classify_github_issues": "td-issues owner/repo [--category C]",
        "list_td_categories": "td-categories",
        "td_split_data": "td-split DATA --output-dir DIR",
        "td_export_onnx": "td-export-onnx --model-name ID -o OUT.onnx",
        "td_train": "td-train DATA --model-name ID --output-dir DIR",
        "read_file": "read-file FILE",
        "list_python_files": "list-files PATH",
        "analyze_code_intelligence": "code-intel PATH [--symbol NAME]",
    }

    defs = {t["function"]["name"]: t["function"]["description"] for t in TOOL_DEFINITIONS_OPENAI}

    for name in TOOL_REGISTRY:
        table.add_row(name, cmd_map.get(name, "—"), defs.get(name, "")[:80])

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[bold]Review controls[/bold] (see [cyan]code-review review --help[/cyan]):\n"
        f"  --check {{{'|'.join(ALL_FAMILIES)}}}  (repeatable family selection)\n"
        "  --td-category NAME             (repeatable; restrict TD categories)\n"
        "  --min-severity LEVEL           (focus on severe findings)\n"
        "  --suggest-fixes / --fix-dry-run / --apply-fixes [-y]  (gated fixes)\n"
        "  --format markdown|json         (saved output format)\n"
        "Web UI: [cyan]code-review-web[/cyan]  (after: uv sync --extra web)\n"
    )


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

@main.command("providers")
@click.pass_context
def providers(ctx):
    """List configured providers, their model/base_url, and key status."""
    from code_review_agent.config import resolve_api_key

    cfg = _load_cfg(ctx.obj.get("config_path"))

    table = Table(title="Configured Providers", show_lines=True)
    table.add_column("Provider", style="green", no_wrap=True)
    table.add_column("Active", justify="center")
    table.add_column("Model", style="cyan")
    table.add_column("Base URL")
    table.add_column("API key", justify="center")

    def key_status(resolved: bool) -> str:
        return "[green]set[/green]" if resolved else "[yellow]missing[/yellow]"

    active = "[green]active[/green]"
    # ollama (no key needed)
    table.add_row(
        "ollama",
        active if cfg.provider == "ollama" else "",
        cfg.ollama.model,
        cfg.ollama.base_url,
        "[dim]n/a[/dim]",
    )
    # openai (generic)
    openai_key = resolve_api_key(cfg.openai.api_key, cfg.openai.api_key_env, "OPENAI_API_KEY")
    table.add_row(
        "openai",
        active if cfg.provider == "openai" else "",
        cfg.openai.model,
        cfg.openai.base_url,
        key_status(bool(openai_key)) + f"\n[dim]{cfg.openai.api_key_env}[/dim]",
    )
    # anthropic
    anth_key = resolve_api_key(cfg.anthropic.api_key, cfg.anthropic.api_key_env, "ANTHROPIC_API_KEY")
    table.add_row(
        "anthropic",
        active if cfg.provider == "anthropic" else "",
        cfg.anthropic.model,
        "[dim](SDK default)[/dim]",
        key_status(bool(anth_key)) + f"\n[dim]{cfg.anthropic.api_key_env}[/dim]",
    )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Active provider: [bold]{cfg.provider}[/bold]  (config: {cfg._source})[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# doctor (health check)
# ---------------------------------------------------------------------------

@main.command("doctor")
@click.pass_context
def doctor(ctx):
    """Health check: detectors, ML runtimes, and the configured LLM backend."""
    from code_review_agent.config import resolve_api_key

    cfg = _load_cfg(ctx.obj.get("config_path"))

    table = Table(title="code-review-agent doctor", show_lines=False)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Detail")

    def ok(msg=""):
        return "[green]OK[/green]", msg

    def fail(msg=""):
        return "[red]FAIL[/red]", msg

    def warn(msg=""):
        return "[yellow]WARN[/yellow]", msg

    rows: list[tuple[str, tuple[str, str]]] = []

    # Use lightweight spec/metadata checks so we never trigger heavy imports
    # (importing tdsuite pulls torch/transformers and can take minutes).
    import importlib.metadata
    import importlib.util

    def _check_installed(label, modname, dist_names):
        if importlib.util.find_spec(modname) is None:
            return (label, fail("not installed"))
        ver = "?"
        for dist in dist_names:
            try:
                ver = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        return (label, ok(f"installed (v{ver})"))

    # --- analyzer package availability + versions ---
    rows.append(_check_installed("ml_code_smell_detector", "ml_code_smell_detector",
                                 ["ml-code-smell-detector", "ml_code_smell_detector"]))
    rows.append(_check_installed("code_quality_analyzer", "code_quality_analyzer",
                                 ["code-quality-analyzer", "code_quality_analyzer"]))
    rows.append(_check_installed("tdsuite", "tdsuite", ["tdsuite"]))

    # --- TD runtimes (spec check only; do not import) ---
    for label, modname, dist in [("torch", "torch", "torch"),
                                  ("onnxruntime", "onnxruntime", "onnxruntime")]:
        if importlib.util.find_spec(modname) is not None:
            try:
                ver = importlib.metadata.version(dist)
            except Exception:
                ver = "?"
            rows.append((f"TD runtime: {label}", ok(f"v{ver}")))
        else:
            rows.append((f"TD runtime: {label}", warn("not available")))

    # --- LLM backend reachability ---
    provider = cfg.provider
    if provider in ("ollama", "openai"):
        if provider == "openai":
            key = resolve_api_key(cfg.openai.api_key, cfg.openai.api_key_env, "OPENAI_API_KEY")
            base = os.environ.get("OPENAI_BASE_URL") or cfg.openai.base_url
            api_key = key or "missing"
        else:
            base = cfg.ollama.base_url
            api_key = cfg.ollama.api_key
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base, api_key=api_key or "none", timeout=8, max_retries=0)
            models = client.models.list()
            n = len(list(models.data))
            rows.append((f"LLM backend ({provider})", ok(f"reachable @ {base} — {n} models")))
        except Exception as e:
            rows.append((f"LLM backend ({provider})", fail(f"{base}: {str(e)[:50]}")))
    else:  # anthropic
        key = resolve_api_key(cfg.anthropic.api_key, cfg.anthropic.api_key_env, "ANTHROPIC_API_KEY")
        if key:
            rows.append(("LLM backend (anthropic)", ok("API key present")))
        else:
            rows.append(("LLM backend (anthropic)", fail("ANTHROPIC_API_KEY not set")))

    # --- MCP extra ---
    if importlib.util.find_spec("mcp") is not None:
        rows.append(("MCP server extra", ok("mcp installed")))
    else:
        rows.append(("MCP server extra", warn("install with: uv sync --extra mcp")))

    for check, (status, detail) in rows:
        table.add_row(check, status, detail)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# ollama-models
# ---------------------------------------------------------------------------

@main.command("ollama-models")
@click.option("--url", default=None, help="Ollama base URL (from config if not set)")
@click.pass_context
def ollama_models(ctx, url):
    """List models available in the local Ollama instance."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    base = url or cfg.ollama.base_url
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base, api_key="ollama")
        models = client.models.list()
        table = Table(title=f"Ollama models @ {base}", show_lines=False)
        table.add_column("Model ID", style="green")
        for m in sorted(models.data, key=lambda x: x.id):
            table.add_row(m.id)
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# Smell-group keys whose entries can be filtered by severity.
_SMELL_GROUP_KEYS = (
    "framework_smells", "huggingface_smells", "general_ml_smells",
    "code_smells", "architectural_smells", "structural_smells",
)


def _sev_rank(value) -> int:
    return _SEVERITY_RANK.get(str(value or "").lower(), _SEVERITY_RANK["medium"])


def _filter_result_by_severity(result: dict, min_severity: str) -> dict:
    """Drop smells below ``min_severity``. Non-dict / errored results pass through.

    Handles both flat smell lists (python smells) and file->smells groupings
    (ml smells). Entries without a severity are treated as 'medium'.
    """
    if not isinstance(result, dict) or "error" in result:
        return result
    threshold = _sev_rank(min_severity)
    filtered = dict(result)
    for key in _SMELL_GROUP_KEYS:
        group = result.get(key)
        if isinstance(group, list):
            new_group = []
            for entry in group:
                if isinstance(entry, dict) and "smells" in entry:
                    kept = [s for s in entry.get("smells", [])
                            if _sev_rank((s or {}).get("severity")) <= threshold]
                    if kept:
                        new_entry = dict(entry)
                        new_entry["smells"] = kept
                        new_group.append(new_entry)
                elif isinstance(entry, dict):
                    if _sev_rank(entry.get("severity")) <= threshold:
                        new_group.append(entry)
            filtered[key] = new_group
    filtered["severity_filter"] = min_severity
    return filtered


def _print_tool_result(result: dict, fmt: str, output: str | None, title: str):
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    json_str = json.dumps(result, default=str, indent=2)

    if output:
        out_path = Path(output)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        console.print(f"[green]Saved to:[/green] {output}")
    elif fmt == "json":
        syntax = Syntax(json_str, "json", theme="monokai")
        console.print(syntax)
    else:
        # Pretty-print summary
        summary = result.get("summary", {})
        if summary:
            table = Table(title=f"[bold]{title} — Summary[/bold]", show_lines=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            for k, v in summary.items():
                table.add_row(k.replace("_", " ").title(), str(v))
            console.print()
            console.print(table)

        # Show findings grouped by severity
        findings_shown = False
        for key in ("framework_smells", "huggingface_smells", "general_ml_smells",
                     "code_smells", "architectural_smells", "structural_smells",
                     "predictions"):
            items = result.get(key, [])
            if not items:
                continue
            findings_shown = True
            console.print(f"\n[bold cyan]{key.replace('_', ' ').title()}[/bold cyan]")
            _render_findings_table(items, key)

        if not findings_shown:
            syntax = Syntax(json.dumps(result, default=str, indent=2)[:4000], "json", theme="monokai")
            console.print(syntax)


def _render_findings_table(items: list, source_key: str):
    if not items:
        return
    # TD predictions
    if source_key == "predictions":
        table = Table(show_lines=False)
        table.add_column("Text", max_width=60)
        table.add_column("Category", style="yellow")
        table.add_column("Confidence", justify="right")
        for p in items:
            if isinstance(p, dict):
                table.add_row(
                    p.get("text", "")[:60],
                    str(p.get("predicted_class", p.get("error", "?"))),
                    f"{p.get('predicted_probability', 0.0):.0%}" if "predicted_probability" in p else "—",
                )
        console.print(table)
        return

    # File→smells grouping
    if isinstance(items, list) and items and isinstance(items[0], dict) and "file" in items[0] and "smells" in items[0]:
        for entry in items:
            console.print(f"  [green]{entry['file']}[/green]")
            for smell in entry.get("smells", []):
                if isinstance(smell, dict):
                    line = smell.get("line_number", "?")
                    col = smell.get("col", "")
                    loc = f":{line}" + (f":{col}" if col else "")
                    name = smell.get("name", "?")
                    console.print(f"    [yellow]{name}[/yellow] @ {loc}")
        return

    # Flat list of smell dicts
    if isinstance(items, list):
        table = Table(show_lines=False)
        table.add_column("Name", style="yellow")
        table.add_column("File")
        table.add_column("Line", justify="right")
        table.add_column("Severity", style="red")
        for item in items[:50]:
            d = item if isinstance(item, dict) else getattr(item, "__dict__", {})
            if isinstance(d, dict):
                table.add_row(
                    str(d.get("name", "?"))[:40],
                    str(d.get("file_path", ""))[-40:],
                    str(d.get("line_number", "?")),
                    str(d.get("severity", "")),
                )
        console.print(table)


def _print_code_intel_table(result: dict, target_path: str):
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    summary = result.get("summary", {})

    # Overview
    console.print(f"\n[bold cyan]Code Intelligence — {target_path}[/bold cyan]\n")
    overview = Table(show_lines=False, show_header=False)
    overview.add_column("Metric", style="dim")
    overview.add_column("Value")
    for k in ("files_analyzed", "total_symbols", "total_functions", "total_classes"):
        if k in summary:
            overview.add_row(k.replace("_", " ").title(), str(summary[k]))
    console.print(overview)

    # Parse errors
    errs = summary.get("parse_errors", {})
    if errs:
        console.print(f"\n[red]Parse errors ({len(errs)} files):[/red]")
        for fp, err in list(errs.items())[:5]:
            console.print(f"  {fp}: [dim]{err}[/dim]")

    # Complexity hotspots
    hotspots = summary.get("complexity_hotspots", [])
    if hotspots:
        table = Table(title="Complexity Hotspots", show_lines=True)
        table.add_column("Function", style="green", no_wrap=True)
        table.add_column("File:Line:Col")
        table.add_column("CC", justify="right", style="red")
        table.add_column("LOC", justify="right")
        table.add_column("Params", justify="right")
        table.add_column("Nesting", justify="right")
        for h in hotspots:
            parent = f"{h['parent_class']}." if h.get("parent_class") else ""
            loc = f"{h['file']}:{h['line']}:{h['col']}"
            cc = str(h["cyclomatic_complexity"])
            if h["cyclomatic_complexity"] >= 10:
                cc = f"[red]{cc}[/red]"
            table.add_row(f"{parent}{h['name']}", loc, cc, str(h["loc"]), str(h["param_count"]), str(h["nesting_depth"]))
        console.print()
        console.print(table)

    # Symbol definitions
    if "symbol_definitions" in result:
        table = Table(title="Symbol Definitions", show_lines=True)
        table.add_column("Name", style="green")
        table.add_column("Kind", style="cyan")
        table.add_column("File:Line:Col")
        table.add_column("Signature")
        for d in result["symbol_definitions"]:
            loc = f"{d['file']}:{d['line']}:{d['col']}"
            table.add_row(d["name"], d["kind"], loc, (d.get("signature") or "")[:50])
        console.print()
        console.print(table)

    # Usages
    if "usages" in result:
        table = Table(title=f"Usages of '{result.get('target', '')}'", show_lines=False)
        table.add_column("File:Line:Col")
        table.add_column("Context")
        for u in result["usages"][:30]:
            loc = f"{u['file']}:{u['line']}:{u['col']}"
            table.add_row(loc, u["context"][:80])
        console.print()
        console.print(table)

    # Import graph
    if "import_graph" in result:
        console.print("\n[bold]Import Graph:[/bold]")
        for fp, edges in list(result["import_graph"].items())[:20]:
            console.print(f"  [green]{fp}[/green]")
            for e in edges[:10]:
                names = ", ".join(e["names"]) if e["names"] else e["module"]
                console.print(f"    → {names}  [dim]:{e['line']}[/dim]")
