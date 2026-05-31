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
- Preferred: connect the bundled stdio MCP server.
  - `uv sync --extra mcp` then run `code-review-mcp`.
  - Tools: `detect_ml_smells`, `detect_python_smells`, `classify_technical_debt`,
    `analyze_code_intelligence`, `list_python_files`, `read_file`.
  - Ready-made configs: [`../integrations/`](../integrations/).
- Fallback: map command execution to a shell tool and report extraction to
  file-read tools.
- Keep retries bounded and evidence-driven.

## Providers

Three providers are available via `--provider` (and `config.yaml`):
- `ollama` (default, local, no key)
- `openai` (generic OpenAI-compatible: OpenAI, Groq, OpenRouter, Together,
  Fireworks, Mistral, llama.cpp, vLLM, LM Studio — set `--base-url`/`--api-key`)
- `anthropic`

Run `code-review providers` and `code-review doctor` to inspect configuration
and environment health.

## Standard Portable Workflow

```bash
uv sync
code-review show-config
code-review providers
code-review run-tool code-intel ./src --top-n 15
code-review run-tool python-smells ./src --type structural
code-review review ./src --output reports/review.md
```

## Validation Checklist

- Commands used are real CLI commands.
- Output includes prioritized findings and location references.
- No secrets are emitted.
- Recommendations are actionable and staged (now/next/later).
