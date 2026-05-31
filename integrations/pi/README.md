# Pi integration

Wire `quality-triage` into Pi either via the **MCP server** or as a generic
**shell tool**. Canonical behaviour lives in [`skills.md`](../../skills.md).

## Option 1 — MCP server

```bash
uv sync --extra mcp
```

Register the stdio MCP server with Pi using its MCP configuration. The server
command is:

```
code-review-mcp
```

or, pinned to this checkout:

```
uv run --directory /abs/path/to/quality-triage code-review-mcp
```

Generic MCP server descriptor (adapt to Pi's MCP config format):

```json
{
  "name": "code-review",
  "transport": "stdio",
  "command": "code-review-mcp",
  "args": []
}
```

Exposed tools: `detect_ml_smells`, `detect_python_smells`,
`classify_technical_debt`, `analyze_code_intelligence`, `list_python_files`,
`read_file`.

## Option 2 — Generic shell tool

Give Pi one shell-execution tool and these deterministic commands:

```bash
uv sync
code-review doctor                                   # environment health check
code-review review <path-or-github-url> --output reports/review.md
code-review run-tool python-smells <path> --type all --format json
code-review run-tool ml-smells <path> --format json
code-review run-tool code-intel <path> --format json
code-review run-tool classify-td --text "TODO: ..." --category general
```

## Behaviour contract
Follow the output contract in [`skills.md`](../../skills.md): critical issues
first with `file:line:col`, then a prioritized now/next/later plan. Keep tool
calls deterministic and idempotent; never invent commands or tool names.
