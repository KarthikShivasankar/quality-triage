"""
Optional Gradio companion for Quality Triage.

The CLI (`code-review review`) is the product. This window runs the same
pipeline. Launch with:  code-review app
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator


def equivalent_cli_review(
    path: str,
    *,
    no_llm: bool = True,
    model: str | None = None,
    tools: list[str] | None = None,
) -> str:
    """CLI command that matches a Gradio review run."""
    from code_review_agent.pipeline import PIPELINE_TOOL_IDS, normalize_pipeline_tools

    parts = ["code-review", "review", (path or "").strip() or "<path>"]
    model = (model or "").strip()
    if model and model not in {"local"}:
        parts += ["--model", model]
    if no_llm:
        parts.append("--no-llm")
    selected = normalize_pipeline_tools(tools)
    if selected != list(PIPELINE_TOOL_IDS):
        for tool in selected:
            parts += ["--tool", tool]
    return " ".join(parts)


def equivalent_cli_tool(tool_name: str, path: str) -> str:
    mapping = {
        "list-files": "list-files",
        "code-intel": "code-intel",
        "python-smells": "python-smells",
        "ml-smells": "ml-smells",
        "td-from-comments": "classify-td",
        "classify-td": "classify-td",
    }
    cmd = mapping.get(tool_name, tool_name)
    if tool_name == "classify-td":
        return 'code-review run-tool classify-td --text "…"'
    if tool_name == "td-from-comments":
        return f"code-review run-tool classify-td --from-file {(path or '').strip() or '<comments.txt>'}"
    return f"code-review run-tool {cmd} {(path or '').strip() or '<path>'}"


def cli_preview_markdown(path: str, model: str, no_llm: bool, tools=None) -> str:
    return (
        "**CLI equivalent:** "
        f"`{equivalent_cli_review(path, no_llm=no_llm, model=model, tools=tools)}`"
    )


def tool_preview_markdown(tool_name: str, path: str) -> str:
    return f"**CLI equivalent:** `{equivalent_cli_tool(tool_name, path)}`"


def _error_markdown(message: str, cli: str) -> str:
    return f"**Could not run review.** {message}\n\nCLI: `{cli}`"


def _archive_choices(output_dir: str) -> list[tuple[str, str]]:
    from code_review_agent.reporter import stored_report_choices

    return stored_report_choices(output_dir)


def _dropdown_update(choices: list[tuple[str, str]], selected: str | None):
    import gradio as gr

    value = selected if selected else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --qt-paper: #EDE6D9;
  --qt-ink: #1F1A14;
  --qt-copper: #A24B2A;
  --qt-graphite: #3D4A52;
  --qt-sage: #4A6B52;
  --qt-amber: #C9892E;
  --qt-rule: #C4B8A5;
}
.gradio-container {
  background: var(--qt-paper) !important;
  font-family: "Source Sans 3", system-ui, sans-serif !important;
  color: var(--qt-ink) !important;
  max-width: 1180px !important;
}
.gradio-container h1, .gradio-container h2 {
  font-family: Fraunces, Georgia, serif !important;
  letter-spacing: -0.02em;
}
button.primary, .primary {
  background: var(--qt-copper) !important;
  border-color: var(--qt-copper) !important;
}
#qt-tools {
  background: #F4EEE4 !important;
  border: 1px solid var(--qt-rule);
  border-radius: 8px;
}
#qt-run-review {
  min-height: 2.6rem;
  font-size: 1.05rem !important;
}
.qt-setup {
  min-width: 280px;
}
#qt-report-md, #qt-archive-md {
  background: #F4EEE4 !important;
  border: 1px solid var(--qt-rule);
  border-radius: 8px;
  padding: 0.4rem 0.2rem 1.2rem;
}
#qt-report-md h1, #qt-archive-md h1,
#qt-report-md .md h1, #qt-archive-md .md h1 {
  font-family: Fraunces, Georgia, serif !important;
  font-size: 1.55rem;
  margin: 0.4rem 0 0.8rem;
}
#qt-report-md h2, #qt-archive-md h2,
#qt-report-md .md h2, #qt-archive-md .md h2 {
  font-family: Fraunces, Georgia, serif !important;
  font-size: 1.2rem;
  margin: 1.2rem 0 0.5rem;
  border-bottom: 1px solid var(--qt-rule);
  padding-bottom: 0.25rem;
}
#qt-report-md table, #qt-archive-md table,
#qt-report-md .md table, #qt-archive-md .md table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.7rem 0 1.1rem;
  font-size: 0.92rem;
}
#qt-report-md th, #qt-archive-md th,
#qt-report-md td, #qt-archive-md td,
#qt-report-md .md th, #qt-archive-md .md th,
#qt-report-md .md td, #qt-archive-md .md td {
  border: 1px solid var(--qt-rule);
  padding: 0.35rem 0.6rem;
  text-align: left;
}
#qt-report-md th, #qt-archive-md th,
#qt-report-md .md th, #qt-archive-md .md th {
  background: #E8DFD0;
}
#qt-report-md pre, #qt-archive-md pre,
#qt-report-md .md pre, #qt-archive-md .md pre {
  background: #E8DFD0;
  padding: 0.75rem 1rem;
  overflow-x: auto;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.85rem;
}
#qt-report-md code, #qt-archive-md code,
#qt-report-md .md code, #qt-archive-md .md code {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.88em;
}
"""


def _theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f6ebe4",
            c100="#ead4c8",
            c200="#d7b09a",
            c300="#c48b6c",
            c400="#b36a46",
            c500="#A24B2A",
            c600="#8a3f24",
            c700="#6f331d",
            c800="#552717",
            c900="#3c1b10",
            c950="#241009",
        ),
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Source Sans 3"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        body_background_fill="#EDE6D9",
        body_text_color="#1F1A14",
        block_background_fill="#F4EEE4",
        block_border_color="#C4B8A5",
        button_primary_background_fill="#A24B2A",
        button_primary_text_color="#F6EBE4",
    )


def _load_cfg(config_path: str | None):
    from code_review_agent.config import get_config, reset_config

    reset_config()
    return get_config(config_path)


def _as_file_list(uploads) -> list[str]:
    if not uploads:
        return []
    if isinstance(uploads, (list, tuple)):
        return [str(u) for u in uploads if u]
    return [str(uploads)]


def _prepare_target(path_text: str, uploads, cfg) -> tuple[str, Any, bool]:
    """Return (review_path, cloned_or_none, delete_when_done)."""
    from code_review_agent.github_utils import resolve_review_target

    files = _as_file_list(uploads)
    if files:
        dest = Path(tempfile.mkdtemp(prefix="qt_upload_"))
        for src in files:
            src_path = Path(src)
            shutil.copy(src_path, dest / src_path.name)
        return str(dest), None, True

    target = (path_text or "").strip()
    if not target:
        raise ValueError("Enter a local path or GitHub URL, or upload Python files.")
    review_path, cloned = resolve_review_target(
        target,
        clone_dir=cfg.github.clone_dir or None,
        depth=cfg.github.depth,
        timeout=cfg.github.timeout,
        persist=False,
        fetch_issues=cfg.github.fetch_issues,
        issue_limit=cfg.github.issue_limit,
    )
    return review_path, cloned, bool(cloned and cloned.is_temp)


def _cleanup(cloned, uploaded_dir: str | None) -> None:
    from code_review_agent.github_utils import cleanup_repo

    if cloned:
        cleanup_repo(cloned)
    if uploaded_dir:
        shutil.rmtree(uploaded_dir, ignore_errors=True)


def run_review(
    path_text: str,
    uploads,
    model: str,
    no_llm: bool,
    extra_context: str,
    config_path: str | None,
    tools: list[str] | None = None,
) -> Iterator[tuple]:
    """Generator so the UI can show detector progress.

    Yields:
      status, report_markdown, findings_rows, json_text, saved_paths,
      archive_choices, selected_report_path
    """
    from code_review_agent.dashboard import findings_table_rows
    from code_review_agent.pipeline import execute_hybrid_review
    from code_review_agent.reporter import (
        ReportRenderer,
        save_report,
        stored_report_choices,
    )

    empty_table: list = []
    cloned = None
    uploaded_dir = None
    output_dir = "./reports"
    try:
        cfg = _load_cfg(config_path)
        output_dir = cfg.report.output_dir
        choices = stored_report_choices(output_dir)
        if not (path_text or "").strip() and not _as_file_list(uploads):
            raise ValueError(
                "Enter a local path or GitHub URL, or upload Python files."
            )
        from code_review_agent.pipeline import normalize_pipeline_tools

        if tools is not None and not normalize_pipeline_tools(tools):
            raise ValueError("Select at least one detector.")
        cli = equivalent_cli_review(path_text, no_llm=no_llm, model=model, tools=tools)
        yield (
            f"Running… `{cli}`",
            "_Running detectors…_",
            empty_table,
            "",
            "",
            choices,
            None,
        )

        review_path, cloned, is_upload = _prepare_target(path_text, uploads, cfg)
        if is_upload and not cloned:
            uploaded_dir = review_path

        steps: list[str] = []

        def on_step(msg: str) -> None:
            steps.append(msg)

        from code_review_agent.github_utils import issue_snippets

        model_arg = (model or "").strip() or None
        result = execute_hybrid_review(
            review_path,
            cfg,
            model=model_arg,
            no_llm=no_llm,
            extra_context=extra_context or "",
            issue_texts=issue_snippets(cloned.issues) if cloned else None,
            tools=tools,
            on_step=on_step,
        )
        note = " · ".join(steps) if steps else "Done."
        if result.synthesis_error:
            note += f" Synthesis failed: {result.synthesis_error}"

        renderer = ReportRenderer(
            include_code_snippets=cfg.report.include_code_snippets,
            max_snippet_lines=cfg.report.max_snippet_lines,
        )
        report_md = renderer.render_markdown(result.report)
        rows = findings_table_rows(result.report)
        json_text = renderer.render_json(result.report)
        written = save_report(
            result.report,
            output_dir=cfg.report.output_dir,
            fmt="archive",
            include_code_snippets=cfg.report.include_code_snippets,
            max_snippet_lines=cfg.report.max_snippet_lines,
        )
        saved = ", ".join(written)
        md_path = next(
            (w for w in written if w.endswith(".md")), written[0] if written else None
        )
        choices = stored_report_choices(output_dir)
        note += f"  \nSaved `{saved}`"
        yield note, report_md, rows, json_text, saved, choices, md_path
    except Exception as exc:
        cli = equivalent_cli_review(path_text, no_llm=no_llm, model=model, tools=tools)
        choices = stored_report_choices(output_dir)
        yield (
            f"Failed: {exc}",
            _error_markdown(str(exc), cli),
            empty_table,
            "",
            "",
            choices,
            None,
        )
    finally:
        _cleanup(cloned, uploaded_dir)


def load_saved_report(path: str) -> tuple[str, str, list, str, str]:
    """Open a stored markdown report for the Results tab."""
    from code_review_agent.dashboard import findings_rows_from_payload
    from code_review_agent.reporter import (
        load_stored_markdown,
        stored_report_payload,
    )

    if not (path or "").strip():
        return "Pick a saved report first.", "_No report selected._", [], "", ""
    try:
        markdown = load_stored_markdown(path)
        payload = stored_report_payload(path)
        rows = findings_rows_from_payload(payload)
        json_text = ""
        json_path = Path(path).with_suffix(".json")
        if json_path.is_file():
            json_text = json_path.read_text(encoding="utf-8")
        status = f"Opened `{path}`"
        return status, markdown, rows, json_text, path
    except Exception as exc:
        return f"Failed: {exc}", f"**Could not open report.** {exc}", [], "", path or ""


def rerun_target_from_report(path: str) -> str:
    from code_review_agent.reporter import target_from_stored

    if not (path or "").strip():
        return ""
    return target_from_stored(path)


def _run_review_ui(
    path_text, uploads, model, no_llm, extra_context, config_path, tools
) -> Iterator[tuple]:
    for chunk in run_review(
        path_text, uploads, model, no_llm, extra_context, config_path, tools
    ):
        status, md, rows, js, saved, choices, selected = chunk
        yield (
            status,
            md,
            rows,
            js,
            saved,
            _dropdown_update(choices, selected),
            md,
            status,
        )


def _refresh_archive(config_path: str | None):
    cfg = _load_cfg(config_path)
    choices = _archive_choices(cfg.report.output_dir)
    selected = choices[0][1] if choices else None
    return _dropdown_update(choices, selected)


def _open_saved_ui(path: str):
    status, md, rows, js, saved = load_saved_report(path)
    return status, md, rows, js, saved, md, status


def run_tool(
    tool_name: str,
    path_text: str,
    uploads,
    td_text: str,
    config_path: str | None,
) -> str:
    from code_review_agent.pipeline import extract_debt_comments
    from code_review_agent.tools import (
        analyze_code_intelligence,
        classify_technical_debt,
        detect_ml_smells,
        detect_python_smells,
        list_python_files,
    )

    cfg = _load_cfg(config_path)
    cloned = None
    uploaded_dir = None
    try:
        if tool_name == "classify-td":
            snippets = [s.strip() for s in (td_text or "").splitlines() if s.strip()]
            if not snippets:
                return json.dumps(
                    {
                        "error": "Paste one snippet per line.",
                        "cli": equivalent_cli_tool("classify-td", path_text),
                    },
                    indent=2,
                )
            return json.dumps(classify_technical_debt(snippets), default=str, indent=2)

        review_path, cloned, is_upload = _prepare_target(path_text, uploads, cfg)
        if is_upload and not cloned:
            uploaded_dir = review_path

        if tool_name == "list-files":
            raw = list_python_files(review_path)
        elif tool_name == "code-intel":
            raw = analyze_code_intelligence(review_path, import_graph=True)
        elif tool_name == "python-smells":
            raw = detect_python_smells(review_path, analysis_type="all")
        elif tool_name == "ml-smells":
            raw = detect_ml_smells(review_path)
        elif tool_name == "td-from-comments":
            from code_review_agent.github_utils import issue_snippets

            texts = extract_debt_comments(review_path, set(cfg.tools.ignore_dirs))
            if cloned:
                texts = texts + issue_snippets(cloned.issues)
            raw = (
                classify_technical_debt(texts)
                if texts
                else {"error": "No TODO/FIXME/HACK comments or GitHub issues found."}
            )
        else:
            raw = {"error": f"Unknown tool: {tool_name}"}
        return json.dumps(raw, default=str, indent=2)[:80_000]
    except Exception as exc:
        return json.dumps(
            {
                "error": str(exc),
                "cli": equivalent_cli_tool(tool_name, path_text),
            },
            indent=2,
        )
    finally:
        _cleanup(cloned, uploaded_dir)


def run_ask(question: str, model: str, config_path: str | None) -> str:
    from code_review_agent.agent import make_agent

    q = (question or "").strip()
    if not q:
        return 'Ask a code-quality question first. CLI: `code-review ask "…"`'
    cfg = _load_cfg(config_path)
    agent = make_agent(cfg, model=(model or "").strip() or None)
    return "".join(agent.ask(q))


def _all_pipeline_tools() -> list[str]:
    from code_review_agent.pipeline import PIPELINE_TOOL_IDS

    return list(PIPELINE_TOOL_IDS)


def _no_pipeline_tools() -> list[str]:
    return []


def build_ui(config_path: str | None = None):
    import gradio as gr

    from code_review_agent.config import DEFAULT_LITELLM_MODEL, get_config
    from code_review_agent.dashboard import FINDINGS_HEADERS
    from code_review_agent.pipeline import pipeline_tool_choices

    cfg = get_config(config_path)
    default_model = cfg.llm.model or DEFAULT_LITELLM_MODEL
    alias_choices = list(cfg.aliases.keys()) + [default_model]

    with gr.Blocks(title="Quality Triage") as demo:
        gr.HTML(
            """
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
               letter-spacing:0.18em;text-transform:uppercase;color:#3D4A52;margin:0">
              Quality Triage · CLI companion
            </p>
            <h1 style="font-family:Fraunces,Georgia,serif;margin:0.2rem 0 0.4rem">
              Same pipeline as the terminal
            </h1>
            <p style="color:#3D4A52;margin:0 0 1rem">
              Tick the detectors you want, then run. The report renders as markdown
              and is saved under <code>reports/</code>. Uncheck “Skip LiteLLM” to
              synthesise with Ollama via LiteLLM.
            </p>
            """,
            padding=False,
        )
        cfg_state = gr.State(config_path)

        with gr.Tabs(elem_id="qt-tabs"):
            with gr.Tab("Review", elem_id="qt-tab-review"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=300, elem_classes=["qt-setup"]):
                        path_in = gr.Textbox(
                            label="Project",
                            placeholder="/path/to/project  or  https://github.com/owner/repo",
                            info="Local folder, GitHub URL, or files below.",
                            elem_id="qt-path",
                        )
                        model_in = gr.Dropdown(
                            label="LiteLLM model",
                            choices=alias_choices,
                            value="local" if "local" in cfg.aliases else default_model,
                            allow_custom_value=True,
                            info="`local` is ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M",
                            elem_id="qt-model",
                        )
                        tool_picks = gr.CheckboxGroup(
                            label="Detectors to run",
                            choices=pipeline_tool_choices(),
                            value=_all_pipeline_tools(),
                            info="At least one required. Default is all.",
                            elem_id="qt-tools",
                        )
                        with gr.Row():
                            all_btn = gr.Button(
                                "Select all", size="sm", elem_id="qt-tools-all"
                            )
                            none_btn = gr.Button(
                                "Clear", size="sm", elem_id="qt-tools-none"
                            )
                        no_llm = gr.Checkbox(
                            label="Skip LiteLLM narrative",
                            value=True,
                            info="On = detectors only (--no-llm). Off = synthesise with the model.",
                            elem_id="qt-no-llm",
                        )
                        with gr.Accordion("More options", open=False):
                            uploads = gr.File(
                                label="Or drop .py files",
                                file_count="multiple",
                                file_types=[".py"],
                                type="filepath",
                                elem_id="qt-uploads",
                            )
                            context_in = gr.Textbox(
                                label="Focus / extra context for synthesis",
                                lines=2,
                                elem_id="qt-context",
                            )
                        cli_preview = gr.Markdown(
                            cli_preview_markdown(
                                "", "local", True, _all_pipeline_tools()
                            ),
                            elem_id="qt-cli-preview",
                        )
                        run_btn = gr.Button(
                            "Run review",
                            variant="primary",
                            elem_id="qt-run-review",
                        )
                    with gr.Column(scale=6, min_width=380):
                        status_md = gr.Markdown(
                            "Ready. Choose detectors, enter a path, then run.",
                            elem_id="qt-status",
                        )
                        with gr.Tabs():
                            with gr.Tab("Report"):
                                report_md = gr.Markdown(
                                    "_No report yet. Run a review or open one in **Results**._",
                                    elem_id="qt-report-md",
                                    elem_classes=["qt-report"],
                                    line_breaks=True,
                                    height=560,
                                )
                            with gr.Tab("Findings"):
                                findings_df = gr.Dataframe(
                                    headers=FINDINGS_HEADERS,
                                    label="Findings",
                                    interactive=False,
                                    wrap=True,
                                    row_count=8,
                                    elem_id="qt-findings",
                                )
                            with gr.Tab("JSON"):
                                json_out = gr.Code(language="json", label="Report JSON")
                            with gr.Tab("Saved files"):
                                saved_out = gr.Textbox(
                                    label="Written paths (markdown, JSON, HTML)",
                                    interactive=False,
                                    elem_id="qt-saved",
                                )

                all_btn.click(fn=_all_pipeline_tools, outputs=tool_picks)
                none_btn.click(fn=_no_pipeline_tools, outputs=tool_picks)
                for _comp in (path_in, model_in, no_llm, tool_picks):
                    _comp.change(
                        fn=cli_preview_markdown,
                        inputs=[path_in, model_in, no_llm, tool_picks],
                        outputs=cli_preview,
                    )

            with gr.Tab("Results", elem_id="qt-tab-results"):
                gr.Markdown(
                    "Every review writes markdown, JSON, and HTML under `reports/`. "
                    "Open a file here to read the rendered markdown, or re-run that target."
                )
                archive_dd = gr.Dropdown(
                    label="Saved reports",
                    choices=_archive_choices(cfg.report.output_dir),
                    value=None,
                    interactive=True,
                    elem_id="qt-archive",
                )
                with gr.Row():
                    refresh_btn = gr.Button(
                        "Refresh list", elem_id="qt-archive-refresh"
                    )
                    open_btn = gr.Button(
                        "View report", variant="primary", elem_id="qt-archive-open"
                    )
                    rerun_btn = gr.Button("Re-run target", elem_id="qt-archive-rerun")
                archive_status = gr.Markdown(
                    "Pick a saved report, then View or Re-run.",
                    elem_id="qt-archive-status",
                )
                archive_md = gr.Markdown(
                    "_No saved report open._",
                    elem_id="qt-archive-md",
                    elem_classes=["qt-report"],
                    line_breaks=True,
                    height=560,
                )
                refresh_btn.click(
                    fn=_refresh_archive,
                    inputs=[cfg_state],
                    outputs=[archive_dd],
                )
                open_btn.click(
                    fn=_open_saved_ui,
                    inputs=[archive_dd],
                    outputs=[
                        archive_status,
                        archive_md,
                        findings_df,
                        json_out,
                        saved_out,
                        report_md,
                        status_md,
                    ],
                )
                rerun_btn.click(
                    fn=rerun_target_from_report,
                    inputs=[archive_dd],
                    outputs=[path_in],
                ).then(
                    fn=_run_review_ui,
                    inputs=[
                        path_in,
                        uploads,
                        model_in,
                        no_llm,
                        context_in,
                        cfg_state,
                        tool_picks,
                    ],
                    outputs=[
                        status_md,
                        report_md,
                        findings_df,
                        json_out,
                        saved_out,
                        archive_dd,
                        archive_md,
                        archive_status,
                    ],
                )

            run_btn.click(
                fn=_run_review_ui,
                inputs=[
                    path_in,
                    uploads,
                    model_in,
                    no_llm,
                    context_in,
                    cfg_state,
                    tool_picks,
                ],
                outputs=[
                    status_md,
                    report_md,
                    findings_df,
                    json_out,
                    saved_out,
                    archive_dd,
                    archive_md,
                    archive_status,
                ],
            )

            with gr.Tab("Tools", elem_id="qt-tab-tools"):
                gr.Markdown(
                    "Same detectors as `code-review run-tool …`. "
                    "Use the CLI for CI and scripting."
                )
                tool_name = gr.Radio(
                    label="Detector",
                    choices=[
                        "list-files",
                        "code-intel",
                        "python-smells",
                        "ml-smells",
                        "td-from-comments",
                        "classify-td",
                    ],
                    value="code-intel",
                    elem_id="qt-tool-name",
                )
                tool_path = gr.Textbox(
                    label="Path or GitHub URL",
                    elem_id="qt-tool-path",
                )
                tool_files = gr.File(
                    label="Or drop .py files",
                    file_count="multiple",
                    file_types=[".py"],
                    type="filepath",
                )
                td_box = gr.Textbox(
                    label="Snippets for classify-td (one per line)",
                    lines=4,
                    placeholder="TODO: split train/test before fit",
                    elem_id="qt-td-text",
                )
                tool_cli = gr.Markdown(
                    tool_preview_markdown("code-intel", ""),
                    elem_id="qt-tool-cli",
                )
                tool_btn = gr.Button(
                    "Run tool", variant="primary", elem_id="qt-run-tool"
                )
                tool_out = gr.Code(
                    language="json", label="Tool JSON", elem_id="qt-tool-out"
                )
                for _comp in (tool_name, tool_path):
                    _comp.change(
                        fn=tool_preview_markdown,
                        inputs=[tool_name, tool_path],
                        outputs=tool_cli,
                    )
                tool_btn.click(
                    fn=run_tool,
                    inputs=[tool_name, tool_path, tool_files, td_box, cfg_state],
                    outputs=[tool_out],
                )

            with gr.Tab("Ask", elem_id="qt-tab-ask"):
                gr.Markdown(
                    "Maps to `code-review ask`. The default LFM2.5 GGUF is "
                    "**completion-only** — use a tools-capable `--model` or stay on Review."
                )
                ask_model = gr.Dropdown(
                    label="Model",
                    choices=alias_choices,
                    value="local" if "local" in cfg.aliases else default_model,
                    allow_custom_value=True,
                    elem_id="qt-ask-model",
                )
                question = gr.Textbox(
                    label="Question",
                    lines=4,
                    placeholder="What does a missing random seed smell look like in sklearn?",
                    elem_id="qt-ask-question",
                )
                ask_btn = gr.Button("Ask", variant="primary", elem_id="qt-ask-run")
                ask_out = gr.Markdown(elem_id="qt-ask-out")
                ask_btn.click(
                    fn=run_ask,
                    inputs=[question, ask_model, cfg_state],
                    outputs=[ask_out],
                )

    return demo


def launch_app(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    config_path: str | None = None,
    inbrowser: bool = True,
) -> None:
    demo = build_ui(config_path=config_path)
    kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "inbrowser": inbrowser,
        "show_error": True,
    }
    try:
        demo.launch(theme=_theme(), css=_CSS, **kwargs)
    except TypeError:
        demo.launch(**kwargs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quality Triage Gradio app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    launch_app(
        host=args.host,
        port=args.port,
        share=args.share,
        config_path=args.config,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
