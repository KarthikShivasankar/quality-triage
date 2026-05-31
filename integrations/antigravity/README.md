# Antigravity integration

Use `quality-triage` from Antigravity via the **MCP server** (recommended) or a
generic **shell tool**. Canonical behaviour: [`skills.md`](../../skills.md).

## Option 1 — MCP server

```bash
uv sync --extra mcp
```

Add an MCP server entry pointing at `code-review-mcp` (stdio transport). Generic
descriptor — adapt to Antigravity's MCP settings UI/JSON:

```json
{
  "mcpServers": {
    "code-review": {
      "command": "code-review-mcp",
      "args": []
    }
  }
}
```

If the console script is not on PATH, pin it to this project:

```json
{
  "mcpServers": {
    "code-review": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/quality-triage", "code-review-mcp"]
    }
  }
}
```

Exposed tools: `detect_ml_smells`, `detect_python_smells`,
`classify_technical_debt`, `analyze_code_intelligence`, `list_python_files`,
`read_file`.

## Option 2 — Generic shell tool

```bash
uv sync
code-review review <path-or-github-url> --output reports/review.md
code-review run-tool python-smells <path> --type all --format json
code-review run-tool ml-smells <path> --format json
code-review run-tool code-intel <path> --format json
```

## Behaviour contract
Report critical issues first with `file:line:col`, then a now/next/later plan
(see [`skills.md`](../../skills.md)). Prefer the smallest command that answers
the question; escalate to a full `review` only when needed.
