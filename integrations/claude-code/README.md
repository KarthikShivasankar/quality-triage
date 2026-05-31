# Claude Code / Claude Desktop integration

Two ways to use `quality-triage` with Claude: the **MCP server** (recommended)
or the **CLI** via shell.

Canonical behaviour: [`skills.md`](../../skills.md) and
[`docs/claude-code.md`](../../docs/claude-code.md). This folder only wires things up.

## Option 1 — MCP server (recommended)

Install the extra and register the server.

```bash
uv sync --extra mcp
```

### Claude Desktop

Merge [`claude_desktop_config.json`](claude_desktop_config.json) into your
Claude Desktop config:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

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

If `code-review-mcp` is not on PATH, use the absolute venv path, e.g. on Windows
`C:\\path\\to\\quality-triage\\.venv\\Scripts\\code-review-mcp.exe`, or run it
through uv:

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

### Claude Code (CLI)

```bash
claude mcp add code-review -- code-review-mcp
# or, pinned to this project:
claude mcp add code-review -- uv run --directory /abs/path/to/quality-triage code-review-mcp
```

Then ask Claude things like: *"Use the code-review tools to find ML smells and
complexity hotspots in `src/`, then give me a prioritized fix list."*

## Option 2 — Skill / CLI

A skill file is provided at [`.claude/skills/code-review/SKILL.md`](.claude/skills/code-review/SKILL.md).
Copy the `.claude/` folder into your project (or `~/.claude/`) so Claude Code
picks up the skill, which drives the `code-review` CLI per `skills.md`.

## Slash-command style prompts

- **Full audit:** "Run `code-review review . --output reports/review.md` and summarize critical issues first, then a now/next/later plan."
- **Fast triage:** "Run `code-review run-tool code-intel .` and `code-review run-tool python-smells . --type structural`; report the top 10 hotspots."
- **Debt scan:** "Classify these TODO/FIXME comments with `code-review run-tool classify-td`."
