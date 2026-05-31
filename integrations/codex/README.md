# Codex integration

Codex works well with `quality-triage` either through the **MCP server** or by
running the `code-review` CLI as shell commands. Codex reads `AGENTS.md` at the
repo root automatically — that file already points at the canonical
[`skills.md`](../../skills.md).

## Option 1 — MCP server (`~/.codex/config.toml`)

Install the extra, then add the server to your Codex config:

```bash
uv sync --extra mcp
```

Append [`config.toml`](config.toml) to `~/.codex/config.toml`:

```toml
[mcp_servers.code-review]
command = "code-review-mcp"
args = []

# If code-review-mcp is not on PATH, pin it to this project via uv:
# command = "uv"
# args = ["run", "--directory", "/abs/path/to/quality-triage", "code-review-mcp"]
```

Codex will then expose `detect_ml_smells`, `detect_python_smells`,
`classify_technical_debt`, `analyze_code_intelligence`, `list_python_files`,
and `read_file` as tools.

## Option 2 — CLI via shell

Codex can run the CLI directly. Keep steps explicit and linear, and persist long
reports with `--output`:

```bash
uv sync
code-review show-config
code-review review . --output reports/review.md
code-review run-tool python-smells . --type all --format json
```

## Codex profile snippet

A reusable profile is provided in [`config.toml`](config.toml). It defines both
the MCP server and a `code-review` profile you can select with
`codex --profile code-review`.

## Notes for Codex agents
- Treat `AGENTS.md` + `skills.md` as the runbook.
- Prefer direct shell execution; avoid implied background state.
- Only select `--provider anthropic`/`openai` when the relevant API key is set
  (run `code-review providers` to check).
