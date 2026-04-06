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
- Python 3.9+
- `uv`

Bootstrap commands:

```bash
uv sync
```

Optional backend setup:
- Ollama (default): no API key required
- Anthropic: set `ANTHROPIC_API_KEY`

Useful config override:
- `CODE_REVIEW_CONFIG=/path/to/config.yaml`

## 4) Core Command Surface

Agent must only use documented commands:

- Full review:
  - `code-review review <path-or-github-url>`
- Ask:
  - `code-review ask "<question>"`
- Single-file deep dive:
  - `code-review analyze-file <file.py>`
- Targeted tool runs:
  - `code-review run-tool ml-smells <path>`
  - `code-review run-tool python-smells <path> --type all`
  - `code-review run-tool classify-td --text "TODO: ..."`
  - `code-review run-tool code-intel <path> --top-n 15`
  - `code-review run-tool list-files <path>`
  - `code-review run-tool read-file <file.py>`
- Interactive mode:
  - `code-review interactive <path-or-github-url>`
- Config/tools discovery:
  - `code-review show-config`
  - `code-review list-tools`
  - `code-review ollama-models`

Provider and model overrides are supported on agent-driven commands:
- `--provider ollama|anthropic`
- `--model <name>`

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
- Use one tool/action for shell command execution.
- Use one tool/action for file reads when presenting report snippets.
- Keep tool calls deterministic and idempotent.
- Preserve safety rules from Section 7.

### Minimal interoperability contract
Any compatible agent should be able to:
1. bootstrap with `uv sync`
2. inspect config with `code-review show-config`
3. execute one full review with `code-review review <target>`
4. run at least one targeted analysis via `code-review run-tool ...`
5. return output using the Section 6 structure
