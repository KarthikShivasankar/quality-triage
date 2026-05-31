---
name: code-review
description: Run the quality-triage code-review agent (ML smells, Python smells, technical-debt classification, AST code intelligence) on a path or GitHub URL. Use when the user asks for a code review, quality audit, smell detection, complexity hotspots, or technical-debt analysis of Python code.
---

# code-review skill

This skill drives the `code-review` CLI / `code-review-mcp` MCP server from the
`quality-triage` project. The canonical contract is `skills.md` at the repo root.

## When to use
- "Review this project / file", "find code smells", "find ML anti-patterns",
  "what's the technical debt here", "show complexity hotspots".

## Setup (once)
```bash
uv sync                 # core
uv sync --extra mcp     # if using the MCP server
```

## Preferred: MCP tools
If the `code-review` MCP server is connected, call its tools directly:
`detect_python_smells`, `detect_ml_smells`, `analyze_code_intelligence`,
`classify_technical_debt`, `list_python_files`, `read_file`.

## CLI fallback
```bash
code-review review <path-or-github-url> --output reports/review.md
code-review run-tool python-smells <path> --type all --format json
code-review run-tool ml-smells <path> --format json
code-review run-tool code-intel <path> --format json
code-review run-tool classify-td --text "TODO: ..." --category security
code-review doctor        # verify environment / backends
code-review providers     # show configured LLM providers
```

## Output contract
1. Executive summary
2. Critical issues first, with `file:line:col`
3. ML-specific issues
4. Code quality & architecture issues
5. Technical debt categories
6. now/next/later remediation plan

Never invent commands, files, or tool names. Prefer the smallest command that
answers the question; escalate to a full `review` only when needed.
