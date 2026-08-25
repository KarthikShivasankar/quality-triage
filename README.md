# Quality Triage — Python quality review CLI

Deterministic Python quality analysis (ML smells, architecture smells, AST intelligence, technical-debt classification) plus optional **LiteLLM** synthesis. Switch between local models and frontier APIs with one `--model` flag.

The **CLI is the product**. Gradio is an optional companion that runs the same pipeline.

Agent contract: [`AGENTS.md`](AGENTS.md)

## Requirements

- Python **≥ 3.10**
- [`uv`](https://docs.astral.sh/uv/)
- Optional: [Ollama](https://ollama.com) (default local backend) or API keys for OpenAI / Anthropic / Groq / Gemini / Azure

```bash
git clone https://github.com/KarthikShivasankar/quality-triage.git
cd quality-triage
uv sync
source .venv/bin/activate
```

Default local model is a small Hugging Face GGUF served by Ollama (~1.7 GB, completion-only — hybrid review, not `--agentic`):

```bash
ollama pull hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
```

LiteLLM string: `ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M` (aliases `local` and `test`).

PyTorch TD backend (only if you need `backend: torch`):

```bash
uv sync --extra td-torch
```

Git-sourced detectors are pinned to commit SHAs in `pyproject.toml`:

- `ml-code-smell-detector` — [ml_smells_detector](https://github.com/KarthikShivasankar/ml_smells_detector)
- `code-quality-analyzer` — [python_smells_detector](https://github.com/KarthikShivasankar/python_smells_detector)
- `tdsuite` — [text_classification](https://github.com/KarthikShivasankar/text_classification)

## Quick start

```bash
# Detectors first, then LiteLLM narrative (TTY dashboard; write the full report)
code-review review ./my_project --output reports/review.md

# Pipeline only — no LLM, JSON out, fail CI on high+ findings
code-review review ./src --no-llm --format json --fail-on high --output reports/review.json

# HTML inspection slip
code-review review ./src --no-llm --format html --output reports/review.html

# GitHub URL (optional /tree/branch/subpath). Clone also fetches recent open issues for TD.
code-review review https://github.com/pypa/sampleproject/tree/main/src --no-llm --format json --output reports/review.json
code-review run-tool list-files https://github.com/pypa/sampleproject/tree/main/src

# Frontier model via LiteLLM
export ANTHROPIC_API_KEY=sk-ant-...
code-review review ./src --model frontier

# Custom OpenAI-compatible server (vLLM, LM Studio, llama.cpp)
code-review review ./src --model openai/my-model --api-base http://127.0.0.1:8000/v1
```

A TTY review prints a health score and top findings. Pass `--output` for the full report; `-v` prints full markdown.

## How review works

Default path is **hybrid**:

1. List Python files
2. AST code intelligence (complexity, imports, symbols)
3. Python code / architecture / structural smells
4. ML anti-patterns
5. Classify TODO/FIXME/HACK comments (GitHub clones also classify up to 10 recent open issues; cap 30)
6. Normalize into structured findings
7. LiteLLM writes executive summary + roadmap from that JSON (skip with `--no-llm`)

`--agentic` restores the older LLM-orchestrated tool loop. Prefer hybrid or `--no-llm` for CI and tests.

## Configuration

All settings live in `config.yaml`. Inspect with `code-review show-config`.

```yaml
llm:
  model: ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
  api_base: http://localhost:11434
  timeout: 120
  max_tokens: 8192
  max_iterations: 20
  temperature: 0
  num_retries: 2
  drop_params: true
  fallbacks: []                  # e.g. [groq/llama-3.3-70b-versatile]

aliases:
  local: ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
  test: ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
  cheap: groq/llama-3.3-70b-versatile
  frontier: anthropic/claude-sonnet-4-6
```

`--model local` / `--model test` / `--model cheap` / `--model frontier` resolve aliases. Hugging Face GGUF tags contain slashes (`hf.co/org/repo:Q4_K_M`) but are **not** LiteLLM prefixes — always write `ollama/hf.co/...`.

This GGUF is **completion-only** (no native tool calling). Use hybrid `review` or `--no-llm`. `--agentic` and `ask` need a tools-capable model.

GitHub clones (`github.fetch_issues`, default on) pull the most recently opened issues (default 10, max 30) into the TD classifier. Set `GITHUB_TOKEN` for private repos and a higher API rate limit.

Smell thresholds stay under `code_smells`, `architectural_smells`, and `structural_smells`.

### Environment variables

Copy [`.env.example`](.env.example). LiteLLM uses provider-native keys:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI and many OpenAI-compatible servers |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `GROQ_API_KEY` | Groq |
| `GEMINI_API_KEY` | Google Gemini |
| `AZURE_API_KEY` / `AZURE_API_BASE` | Azure OpenAI |
| `GITHUB_TOKEN` | Private clones and GitHub issues API |
| `CODE_REVIEW_CONFIG` | Alternate `config.yaml` |
| `QUALITY_TRIAGE_LIVE_LLM` | Enable live LLM tests (`1`) |

## CLI reference

```
code-review [--config PATH] [--ci] [-q|-v] COMMAND [OPTIONS]
```

| Command | Description |
|---|---|
| `review TARGET` | Hybrid review of a local path or GitHub URL |
| `ask QUESTION` | Agentic Q&A with tools |
| `analyze-file FILE` | Pipeline scoped to one Python file |
| `run-tool ml-smells PATH` | ML anti-patterns |
| `run-tool python-smells PATH` | Code / architectural / structural smells |
| `run-tool classify-td --text TEXT` | Technical debt categories |
| `run-tool code-intel PATH` | AST intelligence |
| `run-tool list-files PATH` | List Python files |
| `run-tool read-file FILE` | Read a file with line numbers |
| `interactive TARGET` | Manual tool picker + optional synthesis |
| `show-config` | Print resolved configuration |
| `list-tools` | List analysis tools |
| `models` | Aliases plus local Ollama tags when reachable |
| `app` | Optional Gradio companion (same pipeline) |

Common options: `--model`, `--api-base`, `--fallback`, `--format markdown|json|sarif|html|both`, `--fail-on`, `--no-llm`, `--agentic`, `--output`, `--keep-clone`.

Exit codes: `0` ok, `1` runtime error, `2` findings at or above `--fail-on`.

## Tools

The pipeline (and the agent, when `--agentic` / `ask`) can call:

- **detect_ml_smells** — data leakage, missing seeds, Pandas/NumPy/sklearn/PyTorch/HF misuse
- **detect_python_smells** — long methods, god objects, cycles, cyclomatic complexity, …
- **classify_technical_debt** — 18 TD categories via ONNX (default) or optional torch
- **analyze_code_intelligence** — symbols, metrics, import graph, find-usages (`file:line:col`)
- **read_file** / **list_python_files**

## Optional Gradio companion

Install the UI extra, then launch the same pipeline in a browser. Pipeline-only is on by default (matches `--no-llm`).

```bash
uv sync --extra ui
code-review app
```

`uv sync --group dev` already includes Gradio. The window is a companion, not a second product: prefer `code-review review` for CI and scripting. Reviews are saved under `reports/` as markdown, JSON, and HTML. The **Report** tab renders that markdown; **Results** lists prior runs so you can reopen or re-run them. The default LFM2.5 GGUF is completion-only — do not use the Ask tab or `--agentic` on it.

## Development

```bash
uv sync --group dev
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/code_review_agent --cov-report=term-missing
```

Live LLM tests stay skipped unless `QUALITY_TRIAGE_LIVE_LLM=1` (needs Ollama and the pulled GGUF). HF GGUF tool-calling tests skip unless `QUALITY_TRIAGE_FORCE_TOOLS=1`. GitHub clone E2E needs network and also exercises recent-open-issue fetch. Classify stored issue text with:

```bash
uv run pytest tests/ -v -m e2e
code-review run-tool classify-td --from-file tests/fixtures/pallets_issue_snippets.txt --format json
QUALITY_TRIAGE_LIVE_LLM=1 uv run pytest tests/ -v -m integration
```

## License

MIT
