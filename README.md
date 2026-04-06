# Quality Triage — AI-Powered Code Review Agent

An agentic code review tool that combines ML-based smell detection, AST analysis, and LLM synthesis to produce structured, actionable code quality reports with exact `file:line:col` locations.

## Overview

Quality Triage runs a multi-step agentic loop to review Python codebases:

1. **List files** — maps the project structure
2. **AST code intelligence** — identifies complexity hotspots, cyclomatic complexity, import cycles
3. **Python smell detection** — code, architectural, and structural smells
4. **ML smell detection** — data leakage, reproducibility issues, framework anti-patterns
5. **Technical debt classification** — classifies TODO/FIXME comments into 18 debt categories
6. **LLM synthesis** — generates a structured report with prioritised recommendations

Supports two backends: **Ollama** (default, no API key) and **Anthropic Claude**.

## Features

- Detects **ML anti-patterns**: data leakage, missing random seeds, Pandas/NumPy inefficiencies, PyTorch/TensorFlow misuse, HuggingFace API errors
- Detects **Python code smells**: long methods, large classes, duplicate code, feature envy, cyclic dependencies, god objects, high cyclomatic complexity
- **Technical debt classification** via a transformer model (18 categories)
- **AST code intelligence**: symbol lookup, find-usages, import dependency graph, per-function metrics
- Reviews **local paths or GitHub URLs** (auto-clones, then cleans up)
- Streams output in real time to the terminal
- Saves reports to file with `--output`

## Installation

Requires Python ≥ 3.9 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KarthikShivasankar/quality-triage.git
cd quality-triage

# Install all dependencies (including git-sourced packages)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

> **Note:** Three dependencies are installed from GitHub:
> - `ml-code-smell-detector` — [KarthikShivasankar/ml_smells_detector](https://github.com/KarthikShivasankar/ml_smells_detector)
> - `code-quality-analyzer` — [KarthikShivasankar/python_smells_detector](https://github.com/KarthikShivasankar/python_smells_detector)
> - `tdsuite` — [KarthikShivasankar/text_classification](https://github.com/KarthikShivasankar/text_classification)

## Quick Start

```bash
# Review a local project (uses Ollama by default)
code-review review ./my_project

# Review a GitHub repository
code-review review https://github.com/owner/repo

# Review with Anthropic Claude (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
code-review review ./my_project --provider anthropic

# Save the report to a file
code-review review ./my_project --output report.md
```

## Configuration

All settings live in `config.yaml`. Run `code-review show-config` to inspect the resolved configuration.

```yaml
# Switch provider: ollama (default) or anthropic
provider: ollama

ollama:
  model: qwen2.5-coder:7b      # any Ollama model with function calling
  base_url: http://localhost:11434/v1
  max_tokens: 8192
  max_iterations: 25

anthropic:
  model: claude-opus-4-6
  max_tokens: 8192
  max_iterations: 20
  # api_key: set via ANTHROPIC_API_KEY environment variable

github:
  depth: 1                     # shallow clone
  timeout: 120
  # token: set via GITHUB_TOKEN for private repos
```

Smell detection thresholds (long method line count, cyclomatic complexity, etc.) are fully configurable in `config.yaml` under `code_smells`, `architectural_smells`, and `structural_smells`.

### Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required when using the Anthropic backend |
| `GITHUB_TOKEN` | Optional; used for cloning private repositories |
| `CODE_REVIEW_CONFIG` | Optional path to an alternate `config.yaml` |

## CLI Reference

```
code-review [--config PATH] COMMAND [OPTIONS]
```

### Commands

| Command | Description |
|---|---|
| `review TARGET` | Full AI review of a local path or GitHub URL |
| `ask QUESTION` | Ask the agent a code quality question |
| `analyze-file FILE` | Deep-dive review of a single Python file |
| `run-tool ml-smells PATH` | Detect ML-specific anti-patterns |
| `run-tool python-smells PATH` | Detect code/architectural/structural smells |
| `run-tool classify-td --text TEXT` | Classify text into technical debt categories |
| `run-tool code-intel PATH` | AST code intelligence (symbols, metrics, imports) |
| `run-tool list-files PATH` | List all Python files in a directory |
| `run-tool read-file FILE` | Read a file with line numbers |
| `interactive TARGET` | Interactive tool selector with AI synthesis |
| `show-config` | Print resolved configuration |
| `list-tools` | List all available analysis tools |
| `ollama-models` | List models in local Ollama instance |

### Common Options

| Option | Description |
|---|---|
| `--provider ollama\|anthropic` | Override the LLM backend |
| `--model NAME` | Override the model name |
| `--output FILE` | Save the report to a file |
| `--context TEXT` | Extra context or focus areas for the review |
| `--keep-clone` | Keep the cloned GitHub repo after review |

### Examples

```bash
# Review a specific GitHub branch with Anthropic
code-review review https://github.com/owner/repo/tree/dev --provider anthropic

# Detect ML smells only
code-review run-tool ml-smells ./src

# Detect structural Python smells
code-review run-tool python-smells ./src --type structural

# Classify technical debt from text
code-review run-tool classify-td --text "TODO: fix memory leak" --text "HACK: bypass auth"

# Code intelligence — show complexity hotspots
code-review run-tool code-intel ./src

# Look up where a symbol is defined
code-review run-tool code-intel ./src --symbol MyClass

# Show import dependency graph
code-review run-tool code-intel ./src --imports

# Interactive mode — choose tools, then get AI synthesis
code-review interactive ./my_project

# Use a custom config file
code-review --config /path/to/config.yaml review ./my_project
```

## Tools

The agent has access to six analysis tools:

### `detect_ml_smells`
Wraps `ml_code_smell_detector`. Detects:
- **Framework-specific smells**: Pandas `iterrows`, NumPy chained indexing, Scikit-learn API misuse
- **HuggingFace smells**: improper tokenizer usage, missing padding, model misuse
- **General ML smells**: data leakage, missing random seeds, magic numbers in hyperparameters

### `detect_python_smells`
Wraps `code_quality_analyzer`. Three analysis modes:
- **`code`**: long methods, large classes, duplicate code, feature envy, message chains, data clumps
- **`architectural`**: cyclic dependencies, god objects, hub-like modules, unstable dependencies
- **`structural`**: cyclomatic complexity, depth of inheritance tree, coupling between objects, lack of cohesion

### `classify_technical_debt`
Wraps `tdsuite` (transformer model `KarthikShivasankar/td-classifier`). Classifies text snippets into 18 technical debt categories. Useful for extracting meaning from TODO/FIXME/HACK comments.

### `analyze_code_intelligence`
Pure-Python AST analysis (no external dependency). Returns:
- All class and function definitions with signatures and docstrings
- Per-function metrics: cyclomatic complexity, LOC, nesting depth, parameter count
- Symbol lookup by name with `file:line:col`
- Find all usages of a symbol across the project
- Import dependency graph

### `read_file`
Reads a Python file with line numbers, respecting the `read_file_max_lines` config limit.

### `list_python_files`
Lists all Python files in a directory with sizes, respecting the `ignore_dirs` config.

## Project Structure

```
quality-triage/
├── src/code_review_agent/
│   ├── __init__.py          # Package entry point
│   ├── agent.py             # OllamaAgent, AnthropicAgent, CodeReviewAgent factory
│   ├── cli.py               # Click CLI (review, ask, analyze-file, run-tool, interactive, …)
│   ├── config.py            # YAML loader + typed dataclasses (AppConfig, OllamaConfig, …)
│   ├── code_intel.py        # Pure-Python AST code intelligence
│   ├── github_utils.py      # GitHub URL parsing and git clone helpers
│   ├── prompts.py           # LLM system prompt
│   ├── reporter.py          # Report generation utilities
│   └── tools.py             # Tool implementations + OpenAI/Anthropic schemas
├── config.yaml              # Application configuration
├── pyproject.toml           # Project metadata and dependencies
└── uv.lock                  # Locked dependency versions
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Format code
ruff format src/

# Lint
ruff check src/

# Run tests with coverage
pytest --cov=src/code_review_agent --cov-report=term-missing
```

**Code style:**
- 4-space indentation, max 88-character lines
- Type hints required for all function signatures
- Google-style docstrings for public functions
- Import order: standard library → third-party → local

## Report Structure

Every AI-generated review follows this structure:

1. **Executive Summary** — overall health score (0–100), top 3 priorities
2. **Critical Issues** — must-fix items with code examples
3. **ML-Specific Issues** — data leakage, reproducibility, framework misuse
4. **Code Quality Issues** — smells by category with exact `file:line:col`
5. **Architecture Issues** — module-level problems
6. **Technical Debt** — classified TODO/FIXME snippets
7. **Complexity Hotspots** — top functions by cyclomatic complexity
8. **Improvement Roadmap** — prioritised recommendations with effort estimates

## License

MIT
