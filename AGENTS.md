# Agent Compatibility Guide

This file is a lightweight compatibility entrypoint for AI coding agents.

Canonical behavior lives in:
- [`skills.md`](skills.md)

Claude Code operational details live in:
- [`docs/claude-code.md`](docs/claude-code.md)

Distribution name: **`quality-triage`** (PyPI). Installed CLI command:
**`code-review`**. Import package: `code_review_agent`.

## Quick Agent Start

```bash
# From PyPI:
pip install quality-triage          # or: uv add quality-triage

# From source (dev):
uv sync

code-review show-config
code-review review ./my_project --output reports/review.md

# Optional FastAPI web UI (same tools/agent/fix engine as the CLI):
uv sync --extra web
code-review-web        # http://127.0.0.1:8000  (or: uv run python -m code_review_agent.webapp.app)
```

## Development, Lint & CI

```bash
uv sync --extra web --extra mcp --dev
uv run ruff check src/            # lint (CI gate)
uv run ruff format --check src/   # format (CI gate)
uv run python -m pytest tests/ -q
uv build                          # sdist + wheel
```

CI (`.github/workflows/ci.yml`) runs lint+format, the test suite on Python
3.12/3.13, and a build+`twine check` job. Releases publish to PyPI via
Trusted Publishing (OIDC) on `v*` tags (`.github/workflows/release.yml`); bump
`version` in both `pyproject.toml` and `__init__.py` before tagging. See the
README's *Maintaining the package* section for the full release runbook.

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers config loading, GitHub URL parsing, the pure-Python tool helpers, and per-name catalog smell coverage (`tests/test_smell_coverage.py`). Tests that require optional third-party detectors are skipped automatically when those packages are absent. Current result: **215 passed**.

The agent supplements the installed `code_quality_analyzer` (without editing it) so the whole catalog is reachable: **Lazy Class / Dead Code / Data Class** are invoked explicitly (they are defined upstream but missing from the dispatch list), and **Switch Statements / Deep Inheritance Tree (DIT)** are re-detected by pure-AST helpers in `cqa_supplement.py` (the upstream branch counter and inheritance graph are buggy). Missing thresholds `LAZY_CLASS_LINES` (15) and `DATA_CLASS_METHODS` (5) are supplied by the agent. `Unused Parameters` and `Large Comment Blocks` fold into Speculative Generality and Excessive Comments upstream.

## Supported Command Surface

Use only real CLI commands:
- `code-review review <path-or-github-url>`
  - selection: `--check ml|code|architectural|structural|td|code-intel` (repeatable), `--td-category <name>` (repeatable)
  - reporting: `--min-severity <level>`, `--format markdown|json`
  - fixes (suggest-only by default): `--suggest-fixes`, `--fix-dry-run`, `--apply-fixes [--yes]`
- `code-review ask "<question>"`
- `code-review analyze-file <file.py>` (also: `--check`, `--suggest-fixes`, `--fix-dry-run`, `--apply-fixes`)
- `code-review interactive <path-or-github-url>`
- `code-review run-tool ml-smells <path> [--min-severity <level>]`
- `code-review run-tool python-smells <path> --type all [--min-severity <level>]`
- `code-review run-tool classify-td --text "TODO: ..."`
- `code-review run-tool classify-td-all --text "TODO: ..."`
- `code-review run-tool classify-td-ensemble --category A --category B --text "..."`
- `code-review run-tool td-categories`
- `code-review run-tool td-issues <owner/repo>`
- `code-review run-tool td-split <data.csv> --output-dir <dir>`
- `code-review run-tool td-export-onnx --model-name <id> -o <out.onnx>`
- `code-review run-tool td-train <data.csv> --model-name <id> --output-dir <dir>`
- `code-review run-tool code-intel <path>`
- `code-review run-tool list-files <path>`
- `code-review run-tool read-file <file.py>`
- `code-review show-config`
- `code-review list-tools`
- `code-review providers`
- `code-review doctor`
- `code-review ollama-models`

Provider/model overrides:
- `--provider ollama|openai|anthropic`
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
- Fixes are suggest-only by default. Only `--apply-fixes` (with confirmation, or
  `--yes`) writes files; writes are confined to the target project and create
  `.bak` backups. Never write outside the target project.
