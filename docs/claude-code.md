# Claude Code Playbook

This guide describes reliable execution patterns for using Claude Code with `quality-triage`.

Primary contract:
- [`skills.md`](../skills.md)

Use this document for deeper workflows and troubleshooting.

## Quickstart

```bash
uv sync
code-review show-config
code-review review ./your_project
```

If you want the Anthropic backend:

```bash
export ANTHROPIC_API_KEY=sk-...
code-review review ./your_project --provider anthropic
```

If you want a generic OpenAI-compatible backend (OpenAI, Groq, OpenRouter,
Together, llama.cpp, vLLM, LM Studio):

```bash
export OPENAI_API_KEY=sk-...
code-review review ./your_project --provider openai --model gpt-4o-mini
# Point at another service:
code-review review ./your_project --provider openai \
  --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
```

Check providers and environment health:

```bash
code-review providers   # models + base URLs + API-key status
code-review doctor      # detectors, torch/onnxruntime, LLM backend reachability
```

### MCP server (Claude Code / Claude Desktop)

```bash
uv sync --extra mcp
claude mcp add code-review -- code-review-mcp
```

See [`integrations/claude-code/`](../integrations/claude-code/) for a
`claude_desktop_config.json` snippet and a ready-to-use skill.

## Mode Selection

Choose the smallest command that solves the task:

- Use `review` when you need an end-to-end synthesized report.
- Use `analyze-file` when one Python file is under investigation.
- Use `run-tool` commands for deterministic, scoped checks.
- Use `interactive` when you want selective tool execution before synthesis.

## Recommended Operational Flows

### Flow 1: Standard project review
1. `code-review show-config`
2. `code-review review <target> --output reports/review.md`
3. Extract top 3 critical issues and immediate fixes.

### Flow 2: Triage then escalate
1. `code-review run-tool list-files <target>`
2. `code-review run-tool code-intel <target> --top-n 20`
3. `code-review run-tool python-smells <target> --type structural`
4. Run full review only if hotspots or severe smells appear.

### Flow 3: Focused file failure analysis
1. `code-review analyze-file <path/to/file.py> --output reports/file-review.md`
2. `code-review run-tool code-intel <path/to/file.py> --symbol <SymbolName>`
3. `code-review run-tool python-smells <path/to/file.py> --type code`

### Flow 4: Debt-focused sprint planning
1. Collect TODO/FIXME/HACK snippets.
2. `code-review run-tool classify-td --text "TODO: ..." --text "FIXME: ..."`
3. Group results by debt category and estimated effort.

## Non-Interactive Reliability Tips

- Prefer explicit flags over implicit assumptions (`--provider`, `--model`, `--output`).
- Save long outputs with `--output` so runs are reproducible.
- Run smaller `run-tool` commands before expensive full synthesis when debugging.
- Keep command sequences linear and observable.

## Report Quality Gates

Before accepting agent output, confirm:
- findings contain concrete location references (`file:line` or `file:line:col`)
- critical issues are separated from medium/low items
- recommendations are actionable and prioritized
- unknowns and confidence limits are stated

## Common Failures and Fixes

### `ANTHROPIC_API_KEY not set`
Cause:
- Anthropic provider selected without env var.

Fix:
- Set `ANTHROPIC_API_KEY` or switch to `--provider ollama`.

### `openai provider requires an API key`
Cause:
- `--provider openai` selected without the configured key env var.

Fix:
- Set `OPENAI_API_KEY` (or the `openai.api_key_env` var), pass `--api-key`,
  or switch to `--provider ollama`. Run `code-review providers` to see status.

### No Python files found
Cause:
- target has no `.py` files or ignore patterns exclude too much.

Fix:
- run `code-review run-tool list-files <target>` and adjust directory.

### Tool import errors (`ml_code_smell_detector`, `code_quality_analyzer`, `tdsuite`)
Cause:
- dependencies not installed or environment not synced.

Fix:
- rerun `uv sync` in the project root.

### Slow or noisy full reviews
Cause:
- very large target or broad scope.

Fix:
- use triage flow first (`list-files`, `code-intel`, targeted `python-smells`), then scope review.

## Suggested Claude Code Prompt Patterns

### Fast triage prompt
"Run a structural quality triage on `<target>` using `code-review run-tool code-intel` and `code-review run-tool python-smells --type structural`. Return only top 10 hotspots and why they matter."

### Full audit prompt
"Run `code-review review <target> --output reports/review.md`. Summarize critical issues first, then produce a now/next/later remediation plan."

### File incident prompt
"Run `code-review analyze-file <file.py>`. Focus on reliability and maintainability defects and provide minimal patch recommendations."

## Compatibility Notes

To support non-Claude agents later, keep this hierarchy:
1. `skills.md` = canonical command and behavior contract
2. `docs/claude-code.md` = Claude-specific operating playbook
3. `README.md` = user-facing quickstart and navigation
