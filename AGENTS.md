# Agent Compatibility Guide

This file is a lightweight compatibility entrypoint for AI coding agents.

Canonical behavior lives in:
- [`skills.md`](skills.md)

Claude Code operational details live in:
- [`docs/claude-code.md`](docs/claude-code.md)

## Quick Agent Start

```bash
uv sync
code-review show-config
code-review review ./my_project --output reports/review.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers config loading, GitHub URL parsing, and the pure-Python tool helpers. Tests that require optional third-party detectors are skipped automatically when those packages are absent.

## Supported Command Surface

Use only real CLI commands:
- `code-review review <path-or-github-url>`
- `code-review ask "<question>"`
- `code-review analyze-file <file.py>`
- `code-review interactive <path-or-github-url>`
- `code-review run-tool ml-smells <path>`
- `code-review run-tool python-smells <path> --type all`
- `code-review run-tool classify-td --text "TODO: ..."`
- `code-review run-tool code-intel <path>`
- `code-review run-tool list-files <path>`
- `code-review run-tool read-file <file.py>`
- `code-review show-config`
- `code-review list-tools`
- `code-review ollama-models`

Provider/model overrides:
- `--provider ollama|anthropic`
- `--model <name>`

## Output Expectations

Agent outputs should:
- prioritize critical issues first
- include concrete location references
- provide an actionable remediation plan
- state confidence limits when uncertain

## Safety Rules

- Never expose secrets or API keys.
- Do not run destructive git commands unless explicitly requested.
- Do not revert unrelated local changes.
- Keep workflow deterministic and reproducible.
