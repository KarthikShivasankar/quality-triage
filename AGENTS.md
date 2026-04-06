# Repository Guidelines

AI-powered code review agent using ML smell detection, Python smell detection, and technical debt classification.

## Project Structure

```
Quality_Triage/
├── src/code_review_agent/   # Main package
│   ├── agent.py             # Core review agent logic
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration loader
│   ├── code_intel.py        # Code intelligence analysis
│   ├── github_utils.py      # GitHub cloning utilities
│   ├── prompts.py           # LLM prompt templates
│   ├── reporter.py          # Report generation
│   └── tools.py             # Code analysis tools
├── config.yaml              # Application configuration
├── pyproject.toml           # Project metadata & dependencies
└── uv.lock                  # Dependency lock file
```

## Build, Test, and Development Commands

```bash
# Install dependencies and create virtual environment
uv sync

# Activate environment
source .venv/bin/activate

# Run the CLI
code-review review <path-or-github-url>

# Format code with ruff
ruff format src/

# Lint code
ruff check src/

# Run tests with coverage
pytest --cov=src/code_review_agent
```

## Coding Style & Naming Conventions

- **Indentation**: 4 spaces, no tabs
- **Line length**: Maximum 88 characters
- **Import ordering**: Standard library → third-party → local
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for public functions

**Naming patterns**:
- Modules: `snake_case` (e.g., `code_intel.py`)
- Classes: `CamelCase` (e.g., `CodeReviewAgent`)
- Functions/variables: `snake_case` (e.g., `load_config`)
- Constants: `UPPER_SNAKE_CASE`

**Linting**: Run `ruff check` before committing. Configure in `pyproject.toml`.

## Testing Guidelines

- **Framework**: pytest with pytest-cov for coverage
- **Test location**: Place tests adjacent to source files
- **Test naming**: Prefix with `test_` (e.g., `test_load_config`)
- **Coverage target**: Maintain >80% coverage on new code

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/code_review_agent --cov-report=term-missing
```

## Commit & Pull Request Guidelines

**Commit messages**:
- Use imperative tense: "Add CLI entry point" not "Added CLI entry point"
- Keep subject line under 50 characters
- Reference related issues when applicable

**Pull requests**:
- Include description of changes and motivation
- Link to relevant issues or feature requests
- Ensure all tests pass before submitting
- Update documentation for user-facing changes

## Security & Configuration

**Environment variables**:
- `ANTHROPIC_API_KEY`: Required when using Anthropic Claude backend
- `CODE_REVIEW_CONFIG`: Optional path to alternate config file

**Configuration**:
- All settings in `config.yaml` (Ollama, Anthropic, GitHub, tool thresholds)
- Run `code-review show-config` to inspect resolved configuration
- Never commit API keys; use environment variables instead
