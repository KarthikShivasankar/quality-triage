"""
CLI entry point for code-review-agent. The CLI is the product; Gradio is optional.

Commands:
  review PATH|URL          Hybrid review (pipeline + LiteLLM synthesis)
  ask QUESTION             Ask the agent a question
  analyze-file FILE        Deep-dive on a single file
  run-tool TOOL [opts]     On-demand tool execution
  interactive PATH         Interactive tool selector
  show-config              Print resolved configuration
  list-tools               List all available tools
  models                   Show configured model aliases
  app                      Optional local Gradio companion
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import rich_click as click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FINDINGS = 2

_FORMATS = ["markdown", "json", "sarif", "html", "both"]
_FAIL_ON = ["none", "critical", "high", "medium", "low", "info"]

console = Console()


def _load_cfg(config_path: str | None):
    from code_review_agent.config import get_config, reset_config

    reset_config()
    return get_config(config_path)


def _make_agent(
    cfg, provider_override=None, model_override=None, api_base=None, fallbacks=None
):
    from code_review_agent.agent import make_agent
    from code_review_agent.llm import resolve_llm_model

    model = resolve_llm_model(model=model_override, provider=provider_override, cfg=cfg)
    console.print(f"[dim]Model: [bold]{model}[/bold][/dim]")
    return make_agent(
        cfg,
        model=model_override,
        provider=provider_override,
        api_base=api_base,
        fallbacks=list(fallbacks) if fallbacks else None,
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


def _resolve_target(target: str, cfg, keep_clone: bool = False):
    """Return (review_path, cloned_or_none)."""
    from code_review_agent.github_utils import is_github_url, resolve_review_target

    if is_github_url(target):
        console.print(f"[cyan]Cloning:[/cyan] {target}")
    try:
        review_path, cloned = resolve_review_target(
            target,
            clone_dir=cfg.github.clone_dir or None,
            depth=cfg.github.depth,
            timeout=cfg.github.timeout,
            persist=keep_clone,
            fetch_issues=cfg.github.fetch_issues,
            issue_limit=cfg.github.issue_limit,
        )
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Path does not exist: {target}")
        sys.exit(EXIT_ERROR)
    except Exception as exc:
        label = "Clone failed" if is_github_url(target) else "Error"
        console.print(f"[red]{label}:[/red] {exc}")
        sys.exit(EXIT_ERROR)
    if cloned:
        console.print(
            f"[green]Cloned to:[/green] {cloned.local_path}  "
            f"(commit {cloned.commit_sha[:8]})"
        )
        if cloned.subpath:
            console.print(f"[dim]Subpath:[/dim] {cloned.subpath}")
        if cloned.issues:
            console.print(
                f"[dim]Open issues:[/dim] {len(cloned.issues)} "
                "(most recently opened, used for TD classification)"
            )
    return review_path, cloned


def _cleanup_clone(cloned, keep_clone: bool) -> None:
    if cloned and not keep_clone:
        from code_review_agent.github_utils import cleanup_repo

        cleanup_repo(cloned)
        console.print(f"\n[dim]Cleaned up clone: {cloned.local_path}[/dim]")


@contextmanager
def _resolved_tool_path(ctx, path: str):
    """Yield a local directory for a path or GitHub URL; clean up temp clones."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    review_path, cloned = _resolve_target(path, cfg, keep_clone=False)
    try:
        yield review_path
    finally:
        _cleanup_clone(cloned, keep_clone=False)


def _render_report_text(
    data, fmt: str, include_snippets: bool, max_snippet_lines: int
) -> str:
    from code_review_agent.reporter import ReportRenderer

    renderer = ReportRenderer(
        include_code_snippets=include_snippets,
        max_snippet_lines=max_snippet_lines,
    )
    if fmt == "json":
        return renderer.render_json(data)
    if fmt == "sarif":
        return renderer.render_sarif(data)
    if fmt == "html":
        return renderer.render_html(data)
    return renderer.render_markdown(data)


def _emit_report(
    data, fmt: str, output: str | None, cfg, quiet: bool, verbose: bool = False
) -> list[str]:
    from code_review_agent.dashboard import print_review_dashboard
    from code_review_agent.reporter import save_report

    written: list[str] = []
    include = cfg.report.include_code_snippets
    max_snip = cfg.report.max_snippet_lines
    use_dashboard = (
        not quiet
        and not ctx_ci()
        and fmt in ("markdown", "html", "both")
        and console.is_terminal
    )

    if use_dashboard:
        print_review_dashboard(data, console)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "both":
            md = _render_report_text(data, "markdown", include, max_snip)
            js = _render_report_text(data, "json", include, max_snip)
            md_path = (
                out if out.suffix in {".md", ".markdown"} else out.with_suffix(".md")
            )
            json_path = out.with_suffix(".json")
            md_path.write_text(md, encoding="utf-8")
            json_path.write_text(js, encoding="utf-8")
            written = [str(md_path), str(json_path)]
        else:
            suffix = {
                ".md": "markdown",
                ".json": "json",
                ".sarif": "sarif",
                ".html": "html",
            }.get(out.suffix.lower())
            use_fmt = suffix or fmt
            out.write_text(
                _render_report_text(data, use_fmt, include, max_snip),
                encoding="utf-8",
            )
            written = [str(out)]
        if not quiet:
            console.print(f"[green]Saved to:[/green] {', '.join(written)}")
        return written

    if fmt in ("json", "sarif"):
        click.echo(_render_report_text(data, fmt, include, max_snip))
        return written

    if fmt == "html":
        written = save_report(
            data,
            output_dir=cfg.report.output_dir,
            fmt="archive",
            include_code_snippets=include,
            max_snippet_lines=max_snip,
        )
        if not quiet:
            console.print(f"[green]Saved to:[/green] {', '.join(written)}")
        return written

    if use_dashboard and not verbose:
        if not quiet:
            console.print(
                "\n[dim]Full markdown is saved under reports/ "
                "(pass -v to print it here).[/dim]"
            )
    else:
        console.print(
            _render_report_text(data, "markdown", include, max_snip),
            markup=False,
        )
    written = save_report(
        data,
        output_dir=cfg.report.output_dir,
        fmt="archive",
        include_code_snippets=include,
        max_snippet_lines=max_snip,
    )
    if not quiet:
        console.print(f"[green]Saved to:[/green] {', '.join(written)}")
    return written


def ctx_ci() -> bool:
    """Whether the current console was created for --ci (no color / stderr)."""
    return bool(console.no_color)


def _exit_for_findings(data, fail_on: str) -> None:
    from code_review_agent.reporter import findings_meet_threshold

    if findings_meet_threshold(data.findings, fail_on):
        sys.exit(EXIT_FINDINGS)


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="code-review-agent")
@click.option(
    "--config",
    "-C",
    default=None,
    metavar="PATH",
    help="Path to config.yaml (default: ./config.yaml)",
    is_eager=True,
    expose_value=True,
    envvar="CODE_REVIEW_CONFIG",
)
@click.option("--quiet", "-q", is_flag=True, default=False, help="Minimal output")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose tool progress"
)
@click.option(
    "--ci",
    is_flag=True,
    default=False,
    help="CI mode: no color, --fail-on high unless set",
)
@click.pass_context
def main(ctx, config, quiet, verbose, ci):
    """CLI-first Python quality review.

    Hybrid by default: detectors run first, LiteLLM writes the narrative.
    Switch models with --model (LiteLLM strings or aliases like local / frontier).

    Optional companion UI: code-review app
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["ci"] = ci
    if ci:
        global console
        console = Console(no_color=True, stderr=True)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@main.command()
@click.argument("target")
@click.option("--context", "-c", default="", help="Extra context / focus areas")
@click.option("--output", "-o", default=None, help="Save report to file")
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Legacy provider shim (ollama, anthropic, openai, groq, …)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LiteLLM model or alias (local, cheap, frontier)",
)
@click.option(
    "--api-base", default=None, help="Override api_base (Ollama / vLLM / LM Studio)"
)
@click.option("--fallback", multiple=True, help="Fallback LiteLLM model (repeatable)")
@click.option("--format", "fmt", type=click.Choice(_FORMATS), default=None)
@click.option("--fail-on", type=click.Choice(_FAIL_ON), default=None)
@click.option(
    "--keep-clone", is_flag=True, default=False, help="Keep cloned GitHub repo"
)
@click.option("--no-llm", is_flag=True, default=False, help="Skip LiteLLM synthesis")
@click.option(
    "--tool",
    "tools",
    multiple=True,
    type=click.Choice(
        [
            "list-files",
            "code-intel",
            "python-smells",
            "ml-smells",
            "classify-td",
        ]
    ),
    help="Detector to run (repeatable; default: all)",
)
@click.option(
    "--agentic", is_flag=True, default=False, help="LLM-orchestrated tool loop (legacy)"
)
@click.pass_context
def review(
    ctx,
    target,
    context,
    output,
    provider,
    model,
    api_base,
    fallback,
    fmt,
    fail_on,
    keep_clone,
    no_llm,
    tools,
    agentic,
):
    """Full code review on a local PATH or GitHub URL.

    \b
    Examples:
      code-review review ./my_project
      code-review review ./src --no-llm --format json --fail-on high
      code-review review ./src --model frontier
      code-review review https://github.com/owner/repo/tree/dev/src
    """
    cfg = _load_cfg(ctx.obj.get("config_path"))
    quiet = ctx.obj.get("quiet")
    verbose = ctx.obj.get("verbose")
    ci = ctx.obj.get("ci")
    fmt = fmt or cfg.report.default_format
    fail_on = fail_on or ("high" if ci else cfg.report.fail_on)

    cloned = None
    try:
        review_path, cloned = _resolve_target(target, cfg, keep_clone=keep_clone)
    except Exception as exc:
        console.print(f"[red]Clone failed:[/red] {exc}")
        sys.exit(EXIT_ERROR)

    if not quiet:
        console.print(
            Panel(
                f"[bold cyan]Code Review[/bold cyan]\n"
                f"Target: [green]{target}[/green]\n"
                f"Path:   [dim]{review_path}[/dim]",
                expand=False,
            )
        )
        console.print()

    start = time.time()
    try:
        if agentic:
            agent = _make_agent(cfg, provider, model, api_base, fallback)
            _stream(
                agent,
                agent.review,
                review_path,
                extra_context=context,
                output_path=output,
            )
            console.print(f"\n[dim]Finished in {time.time() - start:.1f}s[/dim]")
            return

        from code_review_agent.github_utils import issue_snippets
        from code_review_agent.pipeline import execute_hybrid_review

        if not quiet:
            console.print("[dim]Running analysis pipeline…[/dim]")
        result = execute_hybrid_review(
            review_path,
            cfg,
            model=model,
            provider=provider,
            api_base=api_base,
            fallbacks=list(fallback) if fallback else None,
            no_llm=no_llm,
            extra_context=context,
            issue_texts=issue_snippets(cloned.issues) if cloned else None,
            tools=list(tools) or None,
            on_step=None if quiet else (lambda msg: console.print(f"[dim]{msg}[/dim]")),
        )
        if result.synthesis_error:
            console.print(
                f"[yellow]Synthesis failed:[/yellow] {result.synthesis_error}"
            )

        _emit_report(result.report, fmt, output, cfg, quiet=quiet, verbose=verbose)
        if not quiet:
            console.print(f"\n[dim]Finished in {time.time() - start:.1f}s[/dim]")
        _exit_for_findings(result.report, fail_on)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted.[/yellow]")
        sys.exit(EXIT_ERROR)
    except Exception as exc:
        console.print(f"[red]Review failed:[/red] {exc}")
        sys.exit(EXIT_ERROR)
    finally:
        _cleanup_clone(cloned, keep_clone)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@main.command()
@click.argument("question")
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", default=None)
@click.option("--model", "-m", default=None)
@click.option("--api-base", default=None)
@click.option("--fallback", multiple=True)
@click.pass_context
def ask(ctx, question, output, provider, model, api_base, fallback):
    """Ask the agent a code quality question."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    agent = _make_agent(cfg, provider, model, api_base, fallback)
    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    _stream(agent, agent.ask, question, output_path=output)


# ---------------------------------------------------------------------------
# analyze-file
# ---------------------------------------------------------------------------


@main.command("analyze-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", default=None)
@click.option("--model", "-m", default=None)
@click.option("--api-base", default=None)
@click.option("--format", "fmt", type=click.Choice(_FORMATS), default=None)
@click.option("--no-llm", is_flag=True, default=False)
@click.option("--fail-on", type=click.Choice(_FAIL_ON), default=None)
@click.pass_context
def analyze_file(
    ctx, file_path, output, provider, model, api_base, fmt, no_llm, fail_on
):
    """Deep-dive review of a single Python file (pipeline + optional synthesis)."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    fmt = fmt or cfg.report.default_format
    fail_on = fail_on or cfg.report.fail_on
    abs_path = str(Path(file_path).resolve())
    quiet = ctx.obj.get("quiet")

    from code_review_agent.pipeline import execute_hybrid_review

    if not quiet:
        console.print(
            Panel(
                f"[bold cyan]File Review[/bold cyan]\n[green]{abs_path}[/green]",
                expand=False,
            )
        )
        console.print()

    result = execute_hybrid_review(
        abs_path,
        cfg,
        model=model,
        provider=provider,
        api_base=api_base,
        no_llm=no_llm,
        parallel=False,
    )
    if result.synthesis_error:
        console.print(f"[yellow]Synthesis failed:[/yellow] {result.synthesis_error}")

    _emit_report(
        result.report,
        fmt,
        output,
        cfg,
        quiet=quiet,
        verbose=ctx.obj.get("verbose"),
    )
    _exit_for_findings(result.report, fail_on)


# ---------------------------------------------------------------------------
# run-tool  (on-demand tool execution)
# ---------------------------------------------------------------------------


@main.group("run-tool")
def run_tool():
    """On-demand execution of individual analysis tools."""


def _tool_or_exit(result: dict, fmt: str, output: str | None, title: str) -> None:
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(EXIT_ERROR)
    _print_tool_result(result, fmt, output, title)


@run_tool.command("ml-smells")
@click.argument("path")
@click.option("--ignore", "-i", multiple=True, help="Dirs to ignore")
@click.option("--output", "-o", default=None, help="Save JSON output to file")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_ml_smells(ctx, path, ignore, output, fmt):
    """Detect ML-specific anti-patterns (data leakage, magic numbers, etc.)."""
    from code_review_agent.tools import detect_ml_smells

    with _resolved_tool_path(ctx, path) as review_path:
        with console.status("[cyan]Running ML smell detector…[/cyan]"):
            result = detect_ml_smells(review_path, ignore_dirs=list(ignore) or None)

    _tool_or_exit(result, fmt, output, "ML Smells")


@run_tool.command("python-smells")
@click.argument("path")
@click.option(
    "--type",
    "analysis_type",
    type=click.Choice(["code", "architectural", "structural", "all"]),
    default="all",
    show_default=True,
)
@click.option("--ignore", "-i", multiple=True)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_python_smells(ctx, path, analysis_type, ignore, output, fmt):
    """Detect code/architectural/structural Python code smells."""
    from code_review_agent.tools import detect_python_smells

    with _resolved_tool_path(ctx, path) as review_path:
        with console.status(
            f"[cyan]Running Python smell detector ({analysis_type})…[/cyan]"
        ):
            result = detect_python_smells(
                review_path,
                analysis_type=analysis_type,
                ignore_dirs=list(ignore) or None,
            )

    _tool_or_exit(result, fmt, output, "Python Smells")


@run_tool.command("classify-td")
@click.option(
    "--text", "-t", multiple=True, help="Text snippet to classify (repeatable)"
)
@click.option(
    "--from-file",
    "from_file",
    type=click.Path(exists=True),
    help="File with one snippet per line",
)
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
        texts += [
            line.strip()
            for line in Path(from_file).read_text().splitlines()
            if line.strip()
        ]
    if not texts:
        console.print("[red]Error:[/red] Provide --text or --from-file")
        sys.exit(EXIT_ERROR)

    with console.status("[cyan]Classifying technical debt…[/cyan]"):
        result = classify_technical_debt(texts, model_path=model_path)

    _tool_or_exit(result, fmt, output, "Technical Debt Classification")


@run_tool.command("code-intel")
@click.argument("path")
@click.option("--symbol", "-s", default=None, help="Look up this symbol")
@click.option("--usages", "-u", default=None, help="Find all usages of this symbol")
@click.option(
    "--metrics", "metrics_only", is_flag=True, help="Show function metrics only"
)
@click.option("--imports", "import_graph", is_flag=True, help="Show import graph")
@click.option("--top-n", default=15, show_default=True)
@click.option("--ignore", "-i", multiple=True)
@click.option("--output", "-o", default=None)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def tool_code_intel(
    ctx, path, symbol, usages, metrics_only, import_graph, top_n, ignore, output, fmt
):
    """AST code intelligence: symbols, metrics, imports, usages."""
    from code_review_agent.tools import analyze_code_intelligence

    with _resolved_tool_path(ctx, path) as review_path:
        with console.status("[cyan]Analyzing code intelligence…[/cyan]"):
            result = analyze_code_intelligence(
                review_path,
                symbol=symbol,
                find_usages_of=usages,
                metrics_only=metrics_only,
                import_graph=import_graph,
                ignore_dirs=list(ignore) or None,
                top_n=top_n,
            )

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        sys.exit(EXIT_ERROR)
    if fmt == "table":
        _print_code_intel_table(result, path)
    else:
        _print_tool_result(result, fmt, output, "Code Intelligence")


@run_tool.command("list-files")
@click.argument("path")
@click.option("--ignore", "-i", multiple=True)
@click.pass_context
def tool_list_files(ctx, path, ignore):
    """List all Python files in a project directory."""
    from code_review_agent.tools import list_python_files

    with _resolved_tool_path(ctx, path) as review_path:
        result = list_python_files(review_path, ignore_dirs=list(ignore) or None)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        sys.exit(EXIT_ERROR)

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
        sys.exit(EXIT_ERROR)

    console.print(
        f"[dim]{result['file']} — {result['shown_lines']}/{result['total_lines']} lines[/dim]\n"
    )
    syntax = Syntax(result["content"], "python", line_numbers=False, theme="monokai")
    console.print(syntax)
    if result.get("truncated"):
        console.print(
            f"\n[yellow]… truncated at {result['shown_lines']} lines[/yellow]"
        )


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


@main.command()
@click.argument("target")
@click.option("--output", "-o", default=None)
@click.option("--provider", "-p", default=None)
@click.option("--model", "-m", default=None)
@click.option("--api-base", default=None)
@click.pass_context
def interactive(ctx, target, output, provider, model, api_base):
    """Interactive tool selector — choose which tools to run, then get AI synthesis."""
    cfg = _load_cfg(ctx.obj.get("config_path"))

    cloned = None
    try:
        review_path, cloned = _resolve_target(target, cfg, keep_clone=False)
    except Exception as exc:
        console.print(f"[red]Clone failed:[/red] {exc}")
        sys.exit(EXIT_ERROR)

    console.print(
        Panel(
            f"[bold cyan]Interactive Code Review[/bold cyan]\nTarget: [green]{target}[/green]",
            expand=False,
        )
    )
    console.print()

    tool_choices = {
        "1": ("List Python files", "list_python_files", {"directory": review_path}),
        "2": (
            "Code Intelligence (AST)",
            "analyze_code_intelligence",
            {"path": review_path},
        ),
        "3": ("Python smells (all)", "detect_python_smells", {"path": review_path}),
        "4": ("ML smells", "detect_ml_smells", {"path": review_path}),
        "5": ("Classify technical debt", None, None),
    }

    console.print("[bold]Available tools:[/bold]")
    for key, (name, _, _) in tool_choices.items():
        console.print(f"  [{key}] {name}")
    console.print("  [a] Run ALL tools")
    console.print("  [q] Quit")
    console.print()

    selected = click.prompt(
        "Select tools (comma-separated, e.g. 1,3,4 or a)", default="a"
    )
    if selected.strip().lower() == "q":
        _cleanup_clone(cloned, False)
        return

    from code_review_agent.pipeline import extract_debt_comments
    from code_review_agent.tools import classify_technical_debt, execute_tool

    keys_to_run = (
        list(tool_choices.keys())
        if selected.strip().lower() == "a"
        else [k.strip() for k in selected.split(",") if k.strip() in tool_choices]
    )

    results: dict[str, Any] = {}

    try:
        for key in keys_to_run:
            name, fn_name, kwargs = tool_choices[key]
            if fn_name is None:
                continue
            with console.status(f"[cyan]Running: {name}…[/cyan]"):
                result = execute_tool(fn_name, kwargs or {})
            results[fn_name] = json.loads(result)
            console.print(f"  [green]✓[/green] {name}")

        if "5" in keys_to_run:
            from code_review_agent.github_utils import issue_snippets

            td_texts = extract_debt_comments(review_path, set(cfg.tools.ignore_dirs))
            if cloned:
                td_texts = td_texts + issue_snippets(cloned.issues)
            if td_texts:
                with console.status("[cyan]Classifying technical debt…[/cyan]"):
                    td_result_raw = classify_technical_debt(td_texts)
                results["classify_technical_debt"] = td_result_raw
                console.print(
                    f"  [green]✓[/green] Technical debt ({len(td_texts)} snippets)"
                )

        console.print()

        if click.confirm("Run AI synthesis of results?", default=True):
            agent = _make_agent(cfg, provider, model, api_base)
            summary = json.dumps(results, default=str, indent=2)[:8000]
            prompt = (
                f"I have run code analysis tools on the project at `{review_path}` and collected these results:\n\n"
                f"```json\n{summary}\n```\n\n"
                "Please synthesise these findings into a structured code review report with:\n"
                "1. Executive summary\n2. Critical issues with exact file:line locations\n"
                "3. Prioritised recommendations\n4. Improvement roadmap"
            )
            console.print()
            _stream(agent, agent.ask, prompt, output_path=output)
    finally:
        _cleanup_clone(cloned, False)


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
            return {
                k: _to_dict(v)
                for k, v in dataclasses.asdict(obj).items()
                if not k.startswith("_")
            }
        return obj

    d = _to_dict(cfg)
    d.pop("_raw", None)
    d.pop("_source", None)
    d["provider"] = cfg.provider

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
    from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI, TOOL_REGISTRY

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

    defs = {
        t["function"]["name"]: t["function"]["description"]
        for t in TOOL_DEFINITIONS_OPENAI
    }

    for name in TOOL_REGISTRY:
        table.add_row(name, cmd_map.get(name, "—"), defs.get(name, "")[:80])

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@main.command("models")
@click.option("--url", default=None, help="Ollama native URL when probing local tags")
@click.pass_context
def models(ctx, url):
    """Show configured LiteLLM aliases and, if reachable, local Ollama tags."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    table = Table(title="Model aliases", show_lines=False)
    table.add_column("Alias", style="cyan")
    table.add_column("LiteLLM model", style="green")
    for alias, model in cfg.aliases.items():
        marker = " (default)" if model == cfg.llm.model else ""
        table.add_row(alias, model + marker)
    console.print()
    console.print(table)
    console.print(f"\n[dim]Active model:[/dim] [bold]{cfg.llm.model}[/bold]")
    if cfg.llm.api_base:
        console.print(f"[dim]api_base:[/dim] {cfg.llm.api_base}")
    console.print(
        "\n[dim]Any LiteLLM string works: ollama/…, openai/…, anthropic/…, groq/…, gemini/…[/dim]"
    )

    base = (url or cfg.llm.api_base or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        import urllib.request

        with urllib.request.urlopen(base + "/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        tags = [m.get("name") for m in data.get("models", []) if m.get("name")]
        if tags:
            otable = Table(title=f"Ollama tags @ {base}", show_lines=False)
            otable.add_column("Model ID", style="green")
            for name in sorted(tags):
                otable.add_row(name)
            console.print()
            console.print(otable)
    except Exception:
        pass
    console.print()


@main.command("ollama-models", hidden=True)
@click.option("--url", default=None)
@click.pass_context
def ollama_models(ctx, url):
    """Deprecated alias for `models`."""
    ctx.invoke(models, url=url)


@main.command("app")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7860, type=int, show_default=True)
@click.option("--share", is_flag=True, default=False, help="Public Gradio tunnel")
@click.option(
    "--no-browser", is_flag=True, default=False, help="Do not open a browser tab"
)
@click.pass_context
def app_cmd(ctx, host, port, share, no_browser):
    """Optional Gradio companion — same pipeline as `code-review review`."""
    try:
        from code_review_agent.app import launch_app
    except ImportError as exc:
        console.print(
            "[red]Gradio is not installed.[/red] "
            "Install the UI extra:\n\n  uv sync --extra ui\n"
        )
        console.print(f"[dim]{exc}[/dim]")
        sys.exit(EXIT_ERROR)
    launch_app(
        host=host,
        port=port,
        share=share,
        config_path=ctx.obj.get("config_path"),
        inbrowser=not no_browser,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_tool_result(result: dict, fmt: str, output: str | None, title: str):
    json_str = json.dumps(result, default=str, indent=2)

    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        console.print(f"[green]Saved to:[/green] {output}")
        return
    if fmt == "json":
        syntax = Syntax(json_str, "json", theme="monokai")
        console.print(syntax)
        return

    summary = result.get("summary", {})
    if summary:
        table = Table(title=f"[bold]{title} — Summary[/bold]", show_lines=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in summary.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print()
        console.print(table)

    findings_shown = False
    for key in (
        "framework_smells",
        "huggingface_smells",
        "general_ml_smells",
        "code_smells",
        "architectural_smells",
        "structural_smells",
        "predictions",
    ):
        items = result.get(key, [])
        if not items:
            continue
        findings_shown = True
        console.print(f"\n[bold cyan]{key.replace('_', ' ').title()}[/bold cyan]")
        _render_findings_table(items, key)

    if not findings_shown and not summary:
        syntax = Syntax(
            json.dumps(result, default=str, indent=2)[:4000], "json", theme="monokai"
        )
        console.print(syntax)


def _render_findings_table(items: list, source_key: str):
    if not items:
        return
    if source_key == "predictions":
        from code_review_agent.reporter import td_class_label

        table = Table(show_lines=False)
        table.add_column("Text", max_width=60)
        table.add_column("Category", style="yellow")
        table.add_column("Confidence", justify="right")
        for p in items:
            if isinstance(p, dict):
                table.add_row(
                    p.get("text", "")[:60],
                    td_class_label(p)
                    if "predicted_class" in p
                    else str(p.get("error", "?")),
                    f"{p.get('predicted_probability', 0.0):.0%}"
                    if "predicted_probability" in p
                    else "—",
                )
        console.print(table)
        return

    if (
        isinstance(items, list)
        and items
        and isinstance(items[0], dict)
        and "file" in items[0]
        and "smells" in items[0]
    ):
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
    summary = result.get("summary", {})

    console.print(f"\n[bold cyan]Code Intelligence — {target_path}[/bold cyan]\n")
    overview = Table(show_lines=False, show_header=False)
    overview.add_column("Metric", style="dim")
    overview.add_column("Value")
    for k in ("files_analyzed", "total_symbols", "total_functions", "total_classes"):
        if k in summary:
            overview.add_row(k.replace("_", " ").title(), str(summary[k]))
    console.print(overview)

    errs = summary.get("parse_errors", {})
    if errs:
        console.print(f"\n[red]Parse errors ({len(errs)} files):[/red]")
        for fp, err in list(errs.items())[:5]:
            console.print(f"  {fp}: [dim]{err}[/dim]")

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
            table.add_row(
                f"{parent}{h['name']}",
                loc,
                cc,
                str(h["loc"]),
                str(h["param_count"]),
                str(h["nesting_depth"]),
            )
        console.print()
        console.print(table)

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

    if "usages" in result:
        table = Table(title="Usages", show_lines=False)
        table.add_column("File:Line:Col")
        table.add_column("Context")
        for u in result["usages"][:30]:
            loc = f"{u['file']}:{u['line']}:{u['col']}"
            table.add_row(loc, u["context"][:80])
        console.print()
        console.print(table)

    if "import_graph" in result:
        console.print("\n[bold]Import Graph:[/bold]")
        for fp, edges in list(result["import_graph"].items())[:20]:
            console.print(f"  [green]{fp}[/green]")
            for e in edges[:10]:
                names = ", ".join(e["names"]) if e["names"] else e["module"]
                console.print(f"    → {names}  [dim]:{e['line']}[/dim]")
