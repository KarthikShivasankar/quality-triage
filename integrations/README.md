# Harness Integrations

Ready-to-use integration assets for popular AI coding harnesses. The
**universal path is the MCP server** (`code-review-mcp`), which exposes all six
analysis tools over stdio. Harnesses without MCP can use the `code-review` CLI
directly (CLI-only fallback).

## The universal path: MCP server

Install the optional extra and you get a stdio MCP server exposing:

`detect_ml_smells`, `detect_python_smells`, `classify_technical_debt`,
`analyze_code_intelligence`, `list_python_files`, `read_file`.

```bash
uv sync --extra mcp          # installs the `mcp` SDK
code-review-mcp              # run the stdio server (console script)
# or: uv run python -m code_review_agent.mcp_server
```

Point any MCP client at the `code-review-mcp` command. Each harness folder below
contains a copy-pasteable config snippet.

## Compatibility matrix

| Harness        | MCP server | CLI fallback | Config file in this folder                         |
|----------------|:----------:|:------------:|----------------------------------------------------|
| Claude Code    | ✅ best    | ✅           | [`claude-code/`](claude-code/)                     |
| Claude Desktop | ✅ best    | —            | [`claude-code/claude_desktop_config.json`](claude-code/claude_desktop_config.json) |
| Codex          | ✅         | ✅ best      | [`codex/`](codex/)                                 |
| Cursor         | ✅         | ✅           | [`claude-code/` MCP snippet works as-is](claude-code/) |
| Pi             | ✅         | ✅           | [`pi/`](pi/)                                        |
| Antigravity    | ✅         | ✅           | [`antigravity/`](antigravity/)                     |

Legend: ✅ = supported, "best" = recommended path for that harness.

## Canonical references

These integration docs are intentionally thin wrappers. The canonical behaviour
lives at the repo root — keep those authoritative and do not duplicate logic:

- [`skills.md`](../skills.md) — agent operating contract
- [`docs/claude-code.md`](../docs/claude-code.md) — Claude Code playbook
- [`docs/agent-interop.md`](../docs/agent-interop.md) — cross-agent portability
- [`AGENTS.md`](../AGENTS.md) — agent compatibility entrypoint

## CLI-only fallback (any harness)

If a harness cannot speak MCP, give it a single shell tool and these commands:

```bash
code-review review <path-or-github-url> --output reports/review.md
code-review run-tool python-smells <path> --type all --format json
code-review run-tool ml-smells <path> --format json
code-review run-tool code-intel <path> --format json
code-review run-tool classify-td --text "TODO: ..." --category security
```
