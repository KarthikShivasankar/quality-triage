# Agent Interoperability Notes

This document helps non-Claude agents execute the same workflows defined in [`skills.md`](../skills.md).

## Canonical Source

Always treat [`skills.md`](../skills.md) as canonical for:
- command surface
- task recipes
- output contract
- safety constraints

This file only explains how to map those rules to other agent runtimes.

## Runtime Mapping

### Codex-style shell agents
- Execute commands exactly as written in `skills.md`.
- Use saved outputs for large reviews:
  - `code-review review <target> --output reports/review.md`
- Avoid inferred tool behavior; keep explicit command steps.

### Generic MCP orchestration agents
- Map command execution to a shell tool.
- Map report extraction to file-read tools.
- Keep retries bounded and evidence-driven.

## Standard Portable Workflow

```bash
uv sync
code-review show-config
code-review run-tool code-intel ./src --top-n 15
code-review run-tool python-smells ./src --type structural
code-review review ./src --output reports/review.md
```

## Validation Checklist

- Commands used are real CLI commands.
- Output includes prioritized findings and location references.
- No secrets are emitted.
- Recommendations are actionable and staged (now/next/later).
