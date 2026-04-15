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

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str | None):
    from code_review_agent.config import get_config, reset_config
    reset_config()
    return get_config(config_path)


def _make_agent(cfg, provider_override=None, model_override=None):
    from code_review_agent.agent import CodeReviewAgent
    provider = provider_override or cfg.provider
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set.")
        console.print("  Use Ollama instead: [cyan]--provider ollama[/cyan]")
        sys.exit(1)
    console.print(
        f"[dim]Provider: [bold]{provider}[/bold]  "
        f"Model: [bold]{model_override or (cfg.ollama.model if provider == 'ollama' else cfg.anthropic.model)}[/bold][/dim]"
    )
    return CodeReviewAgent(
        provider=provider,
        model=model_override if provider == "anthropic" else None,
        ollama_model=model_override if provider == "ollama" else None,
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
    Backends: Ollama (default, no API key needed) or Anthropic Claude.
    Configure everything in config.yaml — run `show-config` to inspect.
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
@click.option("--provider", "-p", type=click.Choice(["ollama", "anthropic"]), default=None)
@click.option("--model", "-m", default=None, help="Override model name")
@click.option("--keep-clone", is_flag=True, default=False, help="Keep cloned GitHub repo")
@click.pass_context
def review(ctx, target, context, output, provider, model, keep_clone):
    """Full AI code review on a local PATH or GitHub URL.

    \b
    Examples:
      code-review review ./my_project
      code-review review https://github.com/owner/repo
      code-review review https://github.com/owner/repo/tree/dev --provider anthropic
    """
    cfg = _load_cfg(ctx.obj.get("config_path"))

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

    agent = _make_agent(cfg, provider, model)

    console.print(Panel(
        f"[bold cyan]Code Review[/bold cyan]\n"
        f"Target: [green]{target}[/green]\n"
        f"Path:   [dim]{review_path}[/dim]",
        expand=False,
    ))
    console.print()

    start = time.time()
    try:
        _stream(agent, agent.review, review_path, extra_context=context, output_path=output)
    finally:
        if cloned and not keep_clone:
            cleanup_repo(cloned)
            console.print(f"\n[dim]Cleaned up clone: {cloned.local_path}[/dim]")

    console.print(f"\n[dim]Finished in {time.time()-start:.1f}s[/dim]")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@main.command()
@click.argument("question")
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", type=click.Choice(["ollama", "anthropic"]), default=None)
@click.option("--model", "-m", default=None)
@click.pass_context
def ask(ctx, question, output, provider, model):
    """Ask the agent a code quality question."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    agent = _make_agent(cfg, provider, model)
    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    _stream(agent, agent.ask, question, output_path=output)


# ---------------------------------------------------------------------------
# analyze-file
# ---------------------------------------------------------------------------

@main.command("analyze-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", type=click.Choice(["ollama", "anthropic"]), default=None)
@click.option("--model", "-m", default=None)
@click.pass_context
def analyze_file(ctx, file_path, output, provider, model):
    """Deep-dive review of a single Python file."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    agent = _make_agent(cfg, provider, model)
    abs_path = str(Path(file_path).resolve())
    prompt = (
        f"Perform a detailed code review of `{abs_path}`. "
        "Read its contents, run analyze_code_intelligence, detect_python_smells, "
        "and detect_ml_smells on it. Provide thorough analysis with exact line:col references."
    )
    console.print(Panel(f"[bold cyan]File Review[/bold cyan]\n[green]{abs_path}[/green]", expand=False))
    console.print()
    _stream(agent, agent.ask, prompt, output_path=output)


# ---------------------------------------------------------------------------
# run-tool  (on-demand tool execution)
# ---------------------------------------------------------------------------

@main.group("run-tool")
def run_tool():
    """On-demand execution of individual analysis tools."""


@run_tool.command("ml-smells")
@click.argument("path", type=click.Path(exists=True))
@click.option("--ignore", "-i", multiple=True, help="Dirs to ignore")
@click.option("--output", "-o", default=None, help="Save JSON output to file")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_ml_smells(ctx, path, ignore, output, fmt):
    """Detect ML-specific anti-patterns (data leakage, magic numbers, etc.)."""
    from code_review_agent.tools import detect_ml_smells
    _load_cfg(ctx.obj.get("config_path"))

    with console.status("[cyan]Running ML smell detector…[/cyan]"):
        result = detect_ml_smells(str(Path(path).resolve()), ignore_dirs=list(ignore) or None)

    _print_tool_result(result, fmt, output, "ML Smells")


@run_tool.command("python-smells")
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", "analysis_type",
              type=click.Choice(["code", "architectural", "structural", "all"]),
              default="all", show_default=True)
@click.option("--ignore", "-i", multiple=True)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_python_smells(ctx, path, analysis_type, ignore, output, fmt):
    """Detect code/architectural/structural Python code smells."""
    from code_review_agent.tools import detect_python_smells
    _load_cfg(ctx.obj.get("config_path"))

    with console.status(f"[cyan]Running Python smell detector ({analysis_type})…[/cyan]"):
        result = detect_python_smells(
            str(Path(path).resolve()),
            analysis_type=analysis_type,
            ignore_dirs=list(ignore) or None,
        )

    _print_tool_result(result, fmt, output, "Python Smells")


@run_tool.command("classify-td")
@click.option("--text", "-t", multiple=True, help="Text snippet to classify (repeatable)")
@click.option("--from-file", "from_file", type=click.Path(exists=True), help="File with one snippet per line")
@click.option("--model-path", default=None, help="HuggingFace model ID override")
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_classify_td(ctx, text, from_file, model_path, output, fmt):
    """Classify text snippets into technical debt categories."""
    from code_review_agent.tools import classify_technical_debt
    _load_cfg(ctx.obj.get("config_path"))

    texts = list(text)
    if from_file:
        texts += [l.strip() for l in Path(from_file).read_text().splitlines() if l.strip()]
    if not texts:
        console.print("[red]Error:[/red] Provide --text or --from-file")
        sys.exit(1)

    with console.status("[cyan]Classifying technical debt…[/cyan]"):
        result = classify_technical_debt(texts, model_path=model_path)

    _print_tool_result(result, fmt, output, "Technical Debt Classification")


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
@click.option("--provider", "-p", type=click.Choice(["ollama", "anthropic"]), default=None)
@click.option("--model", "-m", default=None)
@click.pass_context
def interactive(ctx, target, output, provider, model):
    """Interactive tool selector — choose which tools to run, then get AI synthesis."""
    cfg = _load_cfg(ctx.obj.get("config_path"))

    from code_review_agent.github_utils import is_github_url, clone_repo, cleanup_repo
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
        list_python_files, analyze_code_intelligence,
        detect_python_smells, detect_ml_smells, classify_technical_debt, execute_tool,
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
        agent = _make_agent(cfg, provider, model)
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
        "classify_technical_debt": "classify-td --text TEXT",
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

def _print_tool_result(result: dict, fmt: str, output: str | None, title: str):
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    json_str = json.dumps(result, default=str, indent=2)

    if output:
        Path(output).write_text(json_str, encoding="utf-8")
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
                    p.get("predicted_class", p.get("error", "?")),
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
