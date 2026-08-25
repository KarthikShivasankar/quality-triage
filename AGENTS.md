# AGENTS.md

Operating contract for any coding agent using this repository. This is the only agent-facing file — do not look for per-harness playbooks.

## Purpose

`quality-triage` (`code-review` CLI) reviews Python projects by combining:

1. Deterministic detectors (ML smells, Python smells, AST intelligence, technical-debt classification)
2. Optional LiteLLM synthesis (local Ollama / vLLM / LM Studio, or frontier APIs)

The **CLI is the product**. Gradio (`code-review app`) is an optional companion that runs the same pipeline. Default `review` is **hybrid**: run the pipeline first, then ask the LLM to write a narrative from the structured findings. Do not invent files, tools, or commands that are not listed here.

## Bootstrap

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
code-review show-config
code-review review ./my_project --no-llm --output reports/review.md
```

Default local model is `ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M` (aliases `local`, `test`). Pull it with:

```bash
ollama pull hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
```

That GGUF is completion-only. Prefer hybrid `review` or `--no-llm`. Do not use `--agentic` on it unless the user asked and a tools-capable model is configured.

Optional PyTorch TD backend:

```bash
uv sync --extra td-torch
```

Config override: `CODE_REVIEW_CONFIG=/path/to/config.yaml` or `code-review --config PATH`.

Copy `.env.example` to `.env` and set only the provider keys you use. LiteLLM reads native env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`, `GITHUB_TOKEN`).

## Command surface

Use only these commands:

```bash
code-review review <path-or-github-url>
code-review review <target> --no-llm --format json --fail-on high
code-review review <target> --model frontier
code-review review <target> --model ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
code-review review <target> --format html --output reports/review.html
code-review review <target> --model openai/my-model --api-base http://127.0.0.1:8000/v1
code-review review <target> --agentic
code-review ask "<question>"
code-review analyze-file <file.py>
code-review run-tool ml-smells <path>
code-review run-tool python-smells <path> --type all
code-review run-tool classify-td --text "TODO: ..."
code-review run-tool code-intel <path> --top-n 15
code-review run-tool list-files <path>
code-review run-tool read-file <file.py>
code-review interactive <path-or-github-url>
code-review show-config
code-review list-tools
code-review models
code-review app
```

Flags that matter:

- `--model` — LiteLLM string (`ollama/…`, `openai/…`, `anthropic/…`, `groq/…`, `gemini/…`) or alias (`local`, `cheap`, `frontier`)
- `--api-base` — custom OpenAI-compatible server
- `--fallback` — repeatable fallback model
- `--provider` — legacy shim (`ollama`, `anthropic`, …); prefer `--model`
- `--format markdown|json|sarif|html|both`
- `--fail-on none|critical|high|medium|low|info` (CI: exit 2 when findings meet threshold)
- `--ci` — no color, stderr progress, `--fail-on high` unless overridden
- `--no-llm` — pipeline only
- `--agentic` — LLM chooses tools (non-deterministic)
- `--output` — persist the report
- `--keep-clone` — keep a cloned GitHub repo

Exit codes: `0` ok, `1` runtime error, `2` findings at/above `--fail-on`.

## Recipes

### Full project review

```bash
code-review review <target> --output reports/review.md
```

### CI / no LLM

```bash
code-review review <target> --no-llm --ci --format sarif --output reports/review.sarif
```

### Fast triage then escalate

```bash
code-review run-tool list-files <target>
code-review run-tool code-intel <target> --top-n 20
code-review run-tool python-smells <target> --type structural
```

Then run full `review` only if hotspots appear.

### Single-file incident

```bash
code-review analyze-file <file.py> --output reports/file-review.md
code-review run-tool code-intel <file.py> --symbol <Name>
```

### Technical debt

```bash
code-review run-tool classify-td --text "TODO: ..." --text "FIXME: ..."
code-review run-tool classify-td --from-file tests/fixtures/pallets_issue_snippets.txt --format json
```

GitHub `review` / Gradio import clones fetch the most recently opened issues (default 10, cap 30, or however many exist, including none) and include their title+body in TD classification. Open issues only; pull requests are skipped. API failures are ignored. Disable with `github.fetch_issues: false` in `config.yaml`.

Offline snippets:

```bash
code-review run-tool classify-td --from-file tests/fixtures/pallets_issue_snippets.txt
```

### Optional Gradio companion

```bash
uv sync --extra ui
code-review app
```

`uv sync --group dev` already includes Gradio. The UI runs the same pipeline; prefer the CLI for CI. Pipeline-only is on by default in the UI.

### GitHub URL

```bash
code-review review https://github.com/pypa/sampleproject/tree/main/src --no-llm --format json --output reports/review.json
code-review run-tool list-files https://github.com/pypa/sampleproject/tree/main/src
```

Clone logs `Open issues: N` when any exist. Flask’s open tracker is often empty; Click usually has open issues. Closed issues are not fetched.

### Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/code_review_agent --cov-report=term-missing
uv run pytest tests/ -v -m e2e
QUALITY_TRIAGE_LIVE_LLM=1 uv run pytest tests/ -v -m integration
```

Live LLM tests stay skipped unless `QUALITY_TRIAGE_LIVE_LLM=1`. HF GGUF tool-calling tests skip unless `QUALITY_TRIAGE_FORCE_TOOLS=1`. GitHub clone E2E (`-m e2e`) needs network; skip with `QUALITY_TRIAGE_SKIP_E2E=1`.

## Output contract

Final reviews must include:

1. Executive summary (quality posture; health score when synthesising)
2. Critical issues first, with `file:line` or `file:line:col`
3. ML-specific issues if any
4. Code quality and architecture issues
5. Technical debt categories and examples
6. Prioritized roadmap (now / next / later)

If nothing is found, state what was checked, confidence limits, and residual risk. Never invent locations.

## Safety

- Never expose secrets or API keys.
- Do not run destructive git commands unless explicitly requested.
- Do not revert unrelated local changes.
- Do not claim a command ran when it did not.
- Prefer `--output` for long reports and summarise from the saved artifact.
- Keep runs deterministic: prefer hybrid/`--no-llm` over `--agentic` unless asked.
