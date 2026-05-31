# Skills Contract for AI Agents

This file is the canonical operating contract for running this repository with AI coding agents, optimized for Claude Code first.

## 1) Repository Purpose

`quality-triage` provides an AI-powered Python code review workflow that combines:
- ML smell detection
- Python code/architecture/structural smell detection
- AST code intelligence
- Technical debt text classification
- LLM synthesis into actionable reports

Primary CLI entrypoint:
- `code-review`

## 2) Success Criteria

A successful agent run should:
- produce a readable review with exact `file:line:col` references where possible
- separate critical findings from lower-priority suggestions
- include a practical improvement roadmap
- avoid hallucinated commands, files, or tool names

## 3) Environment Bootstrap

Required:
- Python 3.10+
- `uv`

Bootstrap commands:

```bash
uv sync
```

Optional backend setup (providers):
- `ollama` (default): local, no API key required
- `openai` (generic OpenAI-compatible): set the configured key env var
  (default `OPENAI_API_KEY`). Works with OpenAI, Groq, OpenRouter, Together,
  Fireworks, Mistral, local llama.cpp `server`, vLLM, LM Studio — just change
  `openai.base_url` + `openai.api_key_env` in `config.yaml`.
- `anthropic`: set `ANTHROPIC_API_KEY`

MCP server (universal harness integration), optional extra:

```bash
uv sync --extra mcp
code-review-mcp        # stdio MCP server exposing the 6 analysis tools
```

FastAPI web UI (optional extra) — same tools/agent/fix engine as the CLI:

```bash
uv sync --extra web
code-review-web        # serves http://127.0.0.1:8000 (localhost only)
# or: uv run python -m code_review_agent.webapp.app
```

Fix-application safety contract (CLI `--apply-fixes` and the web Apply action):
- Default is SUGGEST-ONLY; no files are written without explicit opt-in.
- Applying requires confirmation; writes are confined to the target project and
  create `.bak` backups; a fix whose ORIGINAL text no longer matches is skipped.

Useful config override:
- `CODE_REVIEW_CONFIG=/path/to/config.yaml`
- `OPENAI_BASE_URL=...` (overrides `openai.base_url` for the `openai` provider)

## 4) Core Command Surface

Agent must only use documented commands:

- Full review:
  - `code-review review <path-or-github-url>`
  - Select families: `--check ml --check structural` (ml|code|architectural|structural|td|code-intel)
  - Select TD categories: `--td-category security --td-category design`
  - Severity focus / format: `--min-severity high` / `--format markdown|json`
  - Fixes (gated, suggest-only by default): `--suggest-fixes`, `--fix-dry-run`, `--apply-fixes [--yes]`
- Ask:
  - `code-review ask "<question>"`
- Single-file deep dive:
  - `code-review analyze-file <file.py>` (also supports `--check`, `--suggest-fixes`, `--fix-dry-run`, `--apply-fixes`)
- Targeted tool runs:
  - `code-review run-tool ml-smells <path> [--min-severity high]`
  - `code-review run-tool python-smells <path> --type all [--min-severity high]`
  - `code-review run-tool classify-td --text "TODO: ..." [--category security]`
  - `code-review run-tool classify-td-all --text "TODO: ..."`   # sweep all categories
  - `code-review run-tool classify-td-ensemble --category A --category B --text "..."`
  - `code-review run-tool td-categories`                         # list categories + model ids
  - `code-review run-tool td-issues <owner/repo> [--category C]` # GitHub issues TD pipeline
  - `code-review run-tool td-split <data.csv> --output-dir <dir>`
  - `code-review run-tool td-export-onnx --model-name <id> -o <out.onnx>`
  - `code-review run-tool td-train <data.csv> --model-name <id> --output-dir <dir>`
  - `code-review run-tool code-intel <path> --top-n 15`
  - `code-review run-tool list-files <path>`
  - `code-review run-tool read-file <file.py>`
- Interactive mode:
  - `code-review interactive <path-or-github-url>`
- Config/tools discovery:
  - `code-review show-config`
  - `code-review list-tools`
  - `code-review providers`   # configured providers + API-key status
  - `code-review doctor`      # health check: detectors, runtimes, LLM backend
  - `code-review ollama-models`

Provider and model overrides are supported on agent-driven commands
(`review`, `ask`, `analyze-file`, `interactive`):
- `--provider ollama|openai|anthropic`
- `--model <name>`
- `--base-url <url>`   # openai/ollama providers
- `--api-key <key>`    # openai/anthropic providers

Technical-debt classification is now a BINARY, per-category model (no single
18-class model). Pick a category with `--category` (one of 21: general/code/
design/documentation/test/defect/requirement/build/automation/people/process/
infrastructure/architecture/service/security/performance/usability/
maintainability/reliability/portability/compatibility). For a multi-label view
across all categories use `classify-td-all`; for a weighted ensemble use
`classify-td-ensemble` (native torch-free ONNX ensemble by default; add
`--backend torch` to force PyTorch); to classify a repo's GitHub issues use
`td-issues`.

## 5) Agent Task Recipes

### Recipe A: Full project review (default)
1. Run `code-review review <target>`.
2. If user asks for Anthropic, add `--provider anthropic`.
3. If user wants a persisted artifact, add `--output <report-file>`.

### Recipe B: Fast triage before full review
1. Run `code-review run-tool list-files <target>`.
2. Run `code-review run-tool code-intel <target> --top-n 20`.
3. Run `code-review run-tool python-smells <target> --type structural`.
4. Summarize hotspots and ask whether to continue with full review.

### Recipe C: Single-file incident analysis
1. Run `code-review analyze-file <file.py>`.
2. If needed, run:
   - `code-review run-tool code-intel <file.py> --symbol <Name>`
   - `code-review run-tool python-smells <file.py> --type code`
3. Return concrete fixes with file-scoped priorities.

### Recipe D: Technical debt scan
1. Extract or receive TODO/FIXME/HACK snippets.
2. Run repeated `--text` flags:
   - `code-review run-tool classify-td --text "TODO: ..." --text "FIXME: ..."`
3. Group by debt category and impact.

## 6) Output Contract

When generating a final review, agents should follow this shape:
1. Executive summary (overall quality posture)
2. Critical issues first (with location references)
3. ML-specific issues (if any)
4. Code quality and architecture issues
5. Technical debt categories and examples
6. Prioritized roadmap (now/next/later)

If no issues are found, explicitly state:
- what was checked
- confidence limits
- residual risks

## 7) Safety and Reliability Constraints

Agents must:
- never include or invent secrets in outputs
- avoid destructive git operations unless explicitly requested
- avoid claiming commands were run when they were not
- preserve user changes and avoid reverting unrelated work
- prefer deterministic, reproducible command sequences

## 8) Claude Code Usage Notes

Claude Code works best in this repo when:
- command blocks are explicit and copy-pastable
- each run has a clear objective (full review vs targeted smell scan)
- output files are used for long reports (`--output`)
- Anthropic provider is only selected when `ANTHROPIC_API_KEY` is available

## 9) Interop Extension Notes

For other agents, keep this file as canonical and map equivalent workflows:
- command intent remains the same
- provider/model flags remain the same
- output contract remains the same

If another agent needs a dedicated wrapper doc, reference this file rather than duplicating logic.

## 10) Agent Mapping Reference

Use this mapping when porting workflows to other agent ecosystems.

### Codex-style agents
- Treat this file as the system-level runbook.
- Prefer direct shell command execution for `code-review ...` commands.
- Keep execution steps explicit and linear; avoid implied background state.
- Persist long outputs with `--output` and summarize from saved artifacts.

### Generic MCP-compatible agents
- Preferred: connect the bundled MCP server (`code-review-mcp`, install with
  `uv sync --extra mcp`). It exposes 10 analysis tools directly
  (`detect_ml_smells`, `detect_python_smells`, `classify_technical_debt`,
  `classify_technical_debt_all`, `classify_technical_debt_ensemble`,
  `classify_github_issues`, `list_td_categories`, `analyze_code_intelligence`,
  `list_python_files`, `read_file`).

### Code-smell catalog notes
- The agent supplements the installed `code_quality_analyzer` so the full
  catalog is reachable (the package itself is not edited). Now-firing smells
  that were previously dead: **Lazy Class**, **Dead Code**, **Data Class**
  (defined upstream but never dispatched — invoked explicitly per file),
  **Switch Statements** and **Deep Inheritance Tree (DIT)** (re-detected by
  pure-AST supplements in `cqa_supplement.py` because the upstream branch
  counter and inheritance graph are buggy).
- Missing thresholds `LAZY_CLASS_LINES` (15) and `DATA_CLASS_METHODS` (5) are
  supplied by the agent (and `config.yaml`) so those detectors don't `KeyError`.
- `Unused Parameters` and `Large Comment Blocks` are *not* standalone smells
  upstream: they fold into **Speculative Generality** and **Excessive Comments**
  respectively. Coverage is asserted by `tests/test_smell_coverage.py`.
- Ready-made harness configs live under [`integrations/`](integrations/)
  (Claude Code, Codex, Pi, Antigravity).
- Fallback: use one shell tool for `code-review ...` commands and one file-read
  tool for report snippets.
- Keep tool calls deterministic and idempotent.
- Preserve safety rules from Section 7.

### Minimal interoperability contract
Any compatible agent should be able to:
1. bootstrap with `uv sync`
2. inspect config with `code-review show-config`
3. execute one full review with `code-review review <target>`
4. run at least one targeted analysis via `code-review run-tool ...`
5. return output using the Section 6 structure
