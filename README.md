# Quality Triage — AI-Powered Code Review Agent

An agentic code review tool that combines ML-based smell detection, AST analysis,
technical-debt classification, and LLM synthesis to produce structured, actionable
code-quality reports with exact `file:line:col` locations.

Works with **local models (Ollama)**, **any OpenAI-compatible API** (OpenAI,
Groq, OpenRouter, Together, Fireworks, Mistral, llama.cpp, vLLM, LM Studio), or
**Anthropic Claude** — and ships a **universal MCP server** so any MCP-capable
harness (Claude Code, Codex, Cursor, Pi, Antigravity) can drive it.

## Run with an AI coding agent

- Canonical agent contract: `[skills.md](skills.md)`
- Claude-specific playbook: `[docs/claude-code.md](docs/claude-code.md)`
- Cross-agent portability: `[docs/agent-interop.md](docs/agent-interop.md)`
- Harness integration assets (MCP + CLI): `[integrations/](integrations/)`
- Reusable command/prompt snippets: `[scripts/agent_review_examples.md](scripts/agent_review_examples.md)`

Minimal bootstrap:

```bash
uv sync
code-review doctor          # verify environment + backends
code-review review ./my_project --output reports/review.md
```

## Overview

Quality Triage runs a multi-step agentic loop to review Python codebases:

1. **List files** — maps the project structure
2. **AST code intelligence** — complexity hotspots, cyclomatic complexity, import cycles
3. **Python smell detection** — code, architectural, and structural smells
4. **ML smell detection** — data leakage, reproducibility, framework anti-patterns
5. **Technical-debt classification** — binary per-category transformer model
6. **LLM synthesis** — a structured report with prioritized recommendations

## Features

- **Three providers**: `ollama` (default, local, no key), `openai` (any
OpenAI-compatible endpoint, with API keys), `anthropic`.
- **ML anti-patterns**: data leakage, missing random seeds, Pandas/NumPy
inefficiencies, PyTorch/TensorFlow misuse, HuggingFace API errors.
- **Python smells**: long methods, large classes, duplicate code, feature envy,
cyclic dependencies, god objects, high cyclomatic complexity, deep inheritance.
- **Technical-debt classification**: ONNX-first, CPU-friendly, **binary
per-category** model on the HuggingFace Hub (general/code/design/security/…).
- **AST code intelligence**: symbol lookup, find-usages, import graph, per-function metrics.
- **GitHub or local**: auto-clones GitHub URLs (and cleans up), or reviews local paths.
- **Universal MCP server** (`code-review-mcp`) exposing the six tools to any harness.
- Streams output live; saves reports with `--output`.

## Installation

Requires Python ≥ 3.9 and `[uv](https://docs.astral.sh/uv/)`. The MCP extra
requires Python ≥ 3.10.

```bash
git clone https://github.com/KarthikShivasankar/quality-triage.git
cd quality-triage

# Core install (includes git-sourced detectors + onnxruntime)
uv sync

# Optional: MCP server support
uv sync --extra mcp

# Optional: FastAPI web UI
uv sync --extra web
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Or prefix every command with `uv run` (e.g. `uv run code-review doctor`).

> **Three detectors are installed from GitHub (branch `main`):**
>
> - `ml-code-smell-detector` — [KarthikShivasankar/ml_smells_detector](https://github.com/KarthikShivasankar/ml_smells_detector)
> - `code-quality-analyzer` — [KarthikShivasankar/python_smells_detector](https://github.com/KarthikShivasankar/python_smells_detector)
> - `tdsuite` — [KarthikShivasankar/text_classification](https://github.com/KarthikShivasankar/text_classification)

## Quick Start

```bash
# Review a local project (Ollama by default — no key)
code-review review ./my_project

# Review a GitHub repository
code-review review https://github.com/owner/repo

# OpenAI-compatible provider
export OPENAI_API_KEY=sk-...
code-review review ./my_project --provider openai --model gpt-4o-mini

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
code-review review ./my_project --provider anthropic

# Save the report
code-review review ./my_project --output reports/review.md
```

## Providers

Select with `provider:` in `config.yaml` or `--provider` on agent commands.


| Provider    | Key needed | How to configure                                           |
| ----------- | ---------- | ---------------------------------------------------------- |
| `ollama`    | none       | local Ollama at `http://localhost:11434/v1`                |
| `openai`    | yes        | set `openai.base_url` + `openai.api_key_env`; key from env |
| `anthropic` | yes        | `ANTHROPIC_API_KEY`                                        |


The single generic `openai` provider points at any OpenAI-compatible service —
just change the base URL and key:


| Service    | base_url                                | key env (example)    |
| ---------- | --------------------------------------- | -------------------- |
| OpenAI     | `https://api.openai.com/v1`             | `OPENAI_API_KEY`     |
| Groq       | `https://api.groq.com/openai/v1`        | `GROQ_API_KEY`       |
| OpenRouter | `https://openrouter.ai/api/v1`          | `OPENROUTER_API_KEY` |
| Together   | `https://api.together.xyz/v1`           | `TOGETHER_API_KEY`   |
| Fireworks  | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY`  |
| Mistral    | `https://api.mistral.ai/v1`             | `MISTRAL_API_KEY`    |
| llama.cpp  | `http://localhost:8080/v1`              | any string           |
| vLLM       | `http://localhost:8000/v1`              | any string           |
| LM Studio  | `http://localhost:1234/v1`              | any string           |


```bash
# Override base URL / key / model on the fly:
code-review review . --provider openai \
  --base-url https://api.groq.com/openai/v1 --api-key "$GROQ_API_KEY" \
  --model llama-3.3-70b-versatile

# Inspect configured providers + key status (never prints keys):
code-review providers
```

## Technical-debt classification

The TD model family is now a set of **binary, per-category** models on the
HuggingFace Hub (under `karths/...`), inferred ONNX-first on CPU (PyTorch
fallback). Each prediction is `predicted_class` 0/1 (1 = that category of debt is
present) plus a probability — there is no single 18-class model.

```bash
# General technical-debt detector (default)
code-review run-tool classify-td --text "TODO: remove the hard-coded timeout"

# Target a specific category
code-review run-tool classify-td --category security --text "HACK: bypass auth"

# Use a local exported ONNX model (no download)
code-review run-tool classify-td --onnx-path ./model.onnx --text "..."

# Multi-label sweep across every category model
code-review run-tool classify-td-all --text "FIXME: flaky test, no seed set"

# Weighted ensemble of category models (native torch-free ONNX engine by default)
code-review run-tool classify-td-ensemble --text "TODO: hack" \
  --category security --category design --weight 0.6 --weight 0.4

# Force the PyTorch ensemble backend instead of ONNX
code-review run-tool classify-td-ensemble --text "TODO: hack" \
  --category security --category design --backend torch

# Classify a GitHub repo's issues for technical debt
code-review run-tool td-issues pandas-dev/pandas --category defect --limit 30
```

The ensemble runs on tdsuite's native `OnnxEnsembleInferenceEngine` by default —
a torch-free weighted ONNX ensemble (CPU, or GPU with `onnxruntime-gpu`) that
takes the normalised weighted mean of each model's softmax probabilities. Pass
`--backend torch` to use the PyTorch `EnsembleInferenceEngine`; if neither engine
is importable the agent transparently falls back to a per-model CPU ensemble. The
chosen path is reported in the result's `backend` field
(`onnx` / `torch` / `cpu-manual`).

The 21 categories (verified against the live HuggingFace Hub) are: `general`,
`code`, `design`, `documentation`, `test`, `defect`, `requirement`, `build`,
`automation`, `people`, `process`, `infrastructure`, `architecture`, `service`,
`security`, `performance`, `usability`, `maintainability`, `reliability`,
`portability`, `compatibility`. Run `code-review run-tool td-categories` to
list them with their HF model ids. The first run downloads the model (needs
network).

### Dataset / model lifecycle (tdsuite)

The agent also surfaces tdsuite's data and model tooling as `run-tool` commands:

```bash
# Split a labelled dataset into train/test (+ optional top-repo split)
code-review run-tool td-split data.csv --output-dir out/ --repo-column repo

# Export a transformer model to ONNX for CPU inference (needs `onnx`)
code-review run-tool td-export-onnx --model-name karths/binary_classification_train_TD -o model.onnx

# Fine-tune a binary TD classifier (needs torch; GPU strongly recommended)
code-review run-tool td-train data.csv --model-name roberta-base --output-dir model/
```

## Configuration

All settings live in `config.yaml`. Run `code-review show-config` to inspect the
resolved configuration (including the `openai` block).

```yaml
provider: ollama   # ollama | openai | anthropic

ollama:
  model: qwen3.5:4b
  base_url: http://localhost:11434/v1
  max_tokens: 8192
  max_iterations: 25

openai:                              # generic OpenAI-compatible backend
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY        # name of the env var holding the key
  extra_headers: {}                  # e.g. OpenRouter HTTP-Referer / X-Title

anthropic:
  model: claude-opus-4-6
  api_key_env: ANTHROPIC_API_KEY

tools:
  td_classifier:
    model_path: karths/binary_classification_train_TD
    backend: auto                    # auto | onnx | torch
    device: cpu
```

Smell-detection thresholds are configurable under `code_smells`,
`architectural_smells`, and `structural_smells`. Missing keys fall back to the
detector package's own defaults, so partial overrides are safe.

### Environment Variables


| Variable                                                       | Description                                          |
| -------------------------------------------------------------- | ---------------------------------------------------- |
| `OPENAI_API_KEY`                                               | Default key for the `openai` provider                |
| `OPENAI_BASE_URL`                                              | Optional base-URL override for the `openai` provider |
| `GROQ_API_KEY` / `OPENROUTER_API_KEY` / `TOGETHER_API_KEY` / … | Used when `openai.api_key_env` names them            |
| `ANTHROPIC_API_KEY`                                            | Required for the `anthropic` provider                |
| `GITHUB_TOKEN`                                                 | Optional; cloning private repos                      |
| `CODE_REVIEW_CONFIG`                                           | Optional path to an alternate `config.yaml`          |


See `[.env.example](.env.example)` for a template.

## CLI Reference

```
code-review [--config PATH] COMMAND [OPTIONS]
```


| Command                                           | Description                                             |
| ------------------------------------------------- | ------------------------------------------------------- |
| `review TARGET`                                   | Full AI review of a local path or GitHub URL            |
| `ask QUESTION`                                    | Ask the agent a code-quality question                   |
| `analyze-file FILE`                               | Deep-dive review of a single Python file                |
| `interactive TARGET`                              | Interactive tool selector with AI synthesis             |
| `run-tool ml-smells PATH`                         | Detect ML-specific anti-patterns                        |
| `run-tool python-smells PATH [--type all]`        | Detect code/architectural/structural smells             |
| `run-tool classify-td --text TEXT [--category C]` | Binary per-category technical-debt classification       |
| `run-tool classify-td-all --text TEXT`            | Multi-label sweep across every TD category model        |
| `run-tool classify-td-ensemble --category A …`    | Weighted ensemble of TD category models                 |
| `run-tool td-categories`                          | List TD categories + their HuggingFace model ids        |
| `run-tool td-issues owner/repo [--category C]`    | Fetch + classify a repo's GitHub issues for TD          |
| `run-tool td-split DATA --output-dir DIR`         | Split a dataset for TD training (tdsuite)               |
| `run-tool td-export-onnx --model-name ID -o OUT`  | Export a TD model to ONNX for CPU inference             |
| `run-tool td-train DATA --model-name ID -o DIR`   | Fine-tune a binary TD classifier (tdsuite; torch)       |
| `run-tool code-intel PATH`                        | AST code intelligence (symbols, metrics, imports)       |
| `run-tool list-files PATH`                        | List all Python files in a directory                    |
| `run-tool read-file FILE`                         | Read a file with line numbers                           |
| `show-config`                                     | Print resolved configuration                            |
| `list-tools`                                      | List available analysis tools                           |
| `providers`                                       | List providers, models, base URLs, API-key status       |
| `doctor`                                          | Health check: detectors, torch/onnxruntime, LLM backend |
| `ollama-models`                                   | List models in local Ollama instance                    |


### Common Options (agent commands)


| Option                               | Description                         |
| ------------------------------------ | ----------------------------------- |
| `--provider ollama|openai|anthropic` | Override the LLM backend            |
| `--model NAME`                       | Override the model name             |
| `--base-url URL`                     | Override base URL (openai/ollama)   |
| `--api-key KEY`                      | Override API key (openai/anthropic) |
| `--output FILE`                      | Save the report to a file           |
| `--context TEXT`                     | Extra context / focus areas         |
| `--keep-clone`                       | Keep the cloned GitHub repo         |

### Selecting what runs + fixes (`review`, `analyze-file`)

You control which analyses run and whether the agent proposes/applies fixes:

| Option                          | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `--check FAMILY` (repeatable)   | Families to run: `ml`, `code`, `architectural`, `structural`, `td`, `code-intel` |
| `--td-category NAME` (repeat.)  | Restrict TD classification to these categories                           |
| `--min-severity LEVEL`          | Focus the report on findings at/above this severity                      |
| `--format markdown\|json`       | Saved output format (json includes selection + parsed fixes)             |
| `--suggest-fixes`               | Ask the agent for machine-applicable fix blocks (suggest-only)           |
| `--fix-dry-run`                 | Show fix diffs without writing any files                                 |
| `--apply-fixes` [`-y`/`--yes`]  | Apply fixes — gated by confirmation, writes `.bak` backups, never writes outside the target |

The selection restricts the tool schema offered to the LLM **and** the tools it
is allowed to call, so unselected families never run; the report marks them as
*Skipped (not selected)* instead of reporting zero findings.

```bash
# Only ML + structural analysis
code-review review ./proj --check ml --check structural

# TD on chosen categories only
code-review review ./proj --check td --td-category security --td-category design

# Suggest fixes and preview diffs without writing
code-review review ./proj --suggest-fixes --fix-dry-run

# Apply fixes (creates .bak backups; confined to the project)
code-review review ./proj --apply-fixes --yes
```

`run-tool ml-smells` / `run-tool python-smells` also accept `--min-severity`.


## MCP server (universal harness integration)

Install the extra and run the stdio MCP server, which exposes the analysis tools
(`detect_ml_smells`, `detect_python_smells`, `classify_technical_debt`,
`classify_technical_debt_all`, `classify_technical_debt_ensemble`,
`classify_github_issues`, `list_td_categories`, `analyze_code_intelligence`,
`list_python_files`, `read_file`):

```bash
uv sync --extra mcp
code-review-mcp                 # console script
# or: uv run python -m code_review_agent.mcp_server
```

Ready-to-use configs for each harness live in `[integrations/](integrations/)`:


| Harness               | Path                                                     |
| --------------------- | -------------------------------------------------------- |
| Claude Code / Desktop | `[integrations/claude-code/](integrations/claude-code/)` |
| Codex                 | `[integrations/codex/](integrations/codex/)`             |
| Pi                    | `[integrations/pi/](integrations/pi/)`                   |
| Antigravity           | `[integrations/antigravity/](integrations/antigravity/)` |


## Web UI (FastAPI)

A dependency-light, server-rendered HTML UI mirrors the CLI: pick a target, the
provider/model (default Ollama `qwen3.5:4b`), which check families + TD
categories run, toggle fix suggestions, run the review, and view the report,
the selection (with skipped families), and fix diffs — with a gated **Apply
fixes** action. It reuses the same tools, agent, and safe fix engine as the CLI
(no duplicated logic).

```bash
uv sync --extra web
uv run code-review-web          # serves http://127.0.0.1:8000  (localhost only)
# or: uv run python -m code_review_agent.webapp.app
```

Routes: `GET /` (controls), `POST /review` (run + render; `fmt=json` for JSON),
`POST /fixes/apply` (confirmation-gated apply), `GET /healthz`. Override the bind
with `CODE_REVIEW_WEB_HOST` / `CODE_REVIEW_WEB_PORT`.

Harnesses without MCP can use the `code-review` CLI directly (CLI-only fallback,
documented in each folder).

## Tools

The agent (and MCP server) expose six analysis tools:

- `**detect_ml_smells**` — wraps `ml_code_smell_detector`. Framework-specific
(Pandas/NumPy/sklearn/PyTorch/TF), HuggingFace, and general ML smells. A fresh
detector runs per file (no cross-file duplication) and results are normalized
to a canonical shape regardless of detector key differences.
- `**detect_python_smells**` — wraps `code_quality_analyzer`. Modes: `code`
(per-file + cross-file), `architectural` (directory only), `structural`
(directory; single files analyzed via a temp dir). Thresholds always merge
package defaults so detectors never fail on a missing key. See
[Code-smell catalog coverage](#code-smell-catalog-coverage) for the in-repo
fixes that make the full catalog reachable.
- `**classify_technical_debt**` — wraps `tdsuite`, ONNX-first with PyTorch
fallback; binary per-category (21 categories), batch inference. Returns clean
errors instead of crashing if a runtime/model is unavailable. Companions:
`classify_technical_debt_all` (multi-label sweep), `classify_technical_debt_ensemble`
(weighted multi-model on tdsuite's native torch-free `OnnxEnsembleInferenceEngine`
by default; `backend=torch` opt-in; CPU fallback), `classify_github_issues`
(fetch→extract→classify a repo's issues), plus `td_split_data` / `td_export_onnx`
/ `td_train` lifecycle wrappers.
- `**analyze_code_intelligence**` — pure-Python AST: symbols, signatures,
per-function metrics (cyclomatic complexity, LOC, params, nesting), import
graph, find-usages — all with `file:line:col`.
- `**read_file**` — read a file with line numbers (respects `read_file_max_lines`).
- `**list_python_files**` — list `.py` files with sizes (respects `ignore_dirs`).

## Code-smell catalog coverage

The installed `code_quality_analyzer` package (a pinned git dependency we do not
edit) ships several catalog smells that can never fire as-is. The agent closes
these gaps **from within** (`src/code_review_agent/tools.py` +
`src/code_review_agent/cqa_supplement.py`), so the full catalog is reachable.
`tests/test_smell_coverage.py` asserts each one by name.

| Smell | Upstream problem | In-repo fix |
| ----- | ---------------- | ----------- |
| **Lazy Class**, **Dead Code**, **Data Class** | Methods are defined but missing from the `detect_smells` dispatch list, so they never run | `detect_python_smells` explicitly invokes `detect_lazy_class` / `detect_dead_code` / `detect_data_class` per file (via astroid) and merges/de-dupes results |
| **Lazy Class**, **Data Class** thresholds | Reference `LAZY_CLASS_LINES` / `DATA_CLASS_METHODS`, which are absent from the shipped config → `KeyError` | `_AGENT_EXTRA_THRESHOLDS` in `tools.py` supplies defaults (`LAZY_CLASS_LINES: 15`, `DATA_CLASS_METHODS: 5`); also added to `config.yaml` under `code_smells` and overridable there |
| **Switch Statements** | `count_conditions` does not recurse `elif` chains, so the branch count caps at 2 (< threshold 3) and never triggers | Supplemental pure-AST detector (`supplemental_switch_smells`) correctly counts the full `if/elif` chain |
| **Deep Inheritance Tree (DIT)** | Inheritance graph adds edges by bare base name but stores nodes as `module.Class`, so the chain breaks and classes are mislabelled "Isolated" | Supplemental pure-AST detector (`supplemental_dit_smells`) resolves inheritance depth across the analysed files |

> **Notes (documented, not bugs):** `Unused Parameters` and `Large Comment
> Blocks` are not emitted as standalone smells upstream — they are folded into
> **Speculative Generality** (`UNUSED_PARAMETERS_THRESHOLD`) and **Excessive
> Comments** (`LARGE_COMMENT_BLOCKS` / `EXCESSIVE_COMMENTS_RATIO`) respectively.
> Look for those names if you expected the former two.

## Project Structure

```
quality-triage/
├── src/code_review_agent/
│   ├── __init__.py          # package version
│   ├── agent.py             # OpenAICompatibleAgent (ollama+openai), AnthropicAgent, factory
│   ├── cli.py               # Click CLI (review, ask, run-tool, providers, doctor, …)
│   ├── config.py            # YAML loader + dataclasses + threshold/key helpers
│   ├── code_intel.py        # pure-Python AST code intelligence
│   ├── cqa_supplement.py    # in-repo Switch/DIT detectors (fix upstream gaps)
│   ├── github_utils.py      # GitHub URL parsing + git clone
│   ├── mcp_server.py        # stdio MCP server (FastMCP)
│   ├── prompts.py           # LLM system prompt
│   ├── reporter.py          # report generation (markdown/json), binary TD interp
│   └── tools.py             # tool implementations + OpenAI/Anthropic schemas
├── integrations/            # harness wiring (claude-code, codex, pi, antigravity)
├── tests/                   # offline test suite
├── config.yaml              # application configuration
├── pyproject.toml
└── uv.lock
```

## Development

```bash
uv sync --dev
uv run ruff format src/
uv run ruff check src/
uv run python -m pytest tests/ -q
```

## Testing

The suite in `tests/` is **fully offline and deterministic**. LLM calls are
mocked; the real (pure-AST, no-network) detectors run on temp files; the TD
engine is mocked so no model is ever downloaded.


| File                   | Coverage                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `test_config.py`       | loading, defaults, singleton, `flatten_thresholds`, `get_thresholds_flat`, `resolve_api_key`, openai config |
| `test_agent.py`        | provider factory, mocked OpenAI streaming + tool-call loop, `max_tokens`→`max_completion_tokens` retry      |
| `test_tools.py`        | helpers, real Python/ML detectors, ML normalization (both key shapes), mocked TD engine                     |
| `test_td_features.py`  | corrected TD model map, batch/sweep inference, weighted ensemble (native ONNX default, torch opt-in, CPU fallback), GitHub-issues pipeline, data-split/ONNX/train wrappers |
| `test_reporter.py`     | binary TD normalization, ML normalization, markdown/json render                                             |
| `test_code_intel.py`   | symbols, metrics, find-usages, import graph, project summary                                                |
| `test_cli.py`          | `show-config`, `list-tools`, `providers`, `doctor`, `run-tool …`, mocked agent commands                     |
| `test_github_utils.py` | URL detection/parsing, clone cleanup                                                                        |
| `test_smell_coverage.py` | crafted temp files assert each catalog smell fires by name through the wrappers (incl. the formerly-dead ones) |


Run it:

```bash
uv run python -m pytest tests/ -q
```

Current result: **215 passed** in ~24 s (all detectors installed; web tests run when the `web` extra is present).

## Report Structure

Every AI-generated review follows:

1. **Executive Summary** — health score, top 3 priorities
2. **Critical Issues** — must-fix items with code examples
3. **ML-Specific Issues** — data leakage, reproducibility, framework misuse
4. **Code Quality Issues** — smells by category with `file:line:col`
5. **Architecture Issues** — module-level problems
6. **Technical Debt** — classified snippets
7. **Complexity Hotspots** — top functions by cyclomatic complexity
8. **Improvement Roadmap** — prioritized now/next/later plan

## License

MIT