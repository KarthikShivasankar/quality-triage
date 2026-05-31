# Changelog

All notable changes to **quality-triage** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-05-31

### Changed

- **Raised the supported Python floor to 3.12** (`requires-python = ">=3.12"`).
  The bundled detector engines (git `main`) use PEP 701 multi-line f-strings
  (3.12+), and the pinned `onnxruntime` ships no wheels for 3.10, so 3.10/3.11
  could never run the dev/CI dependency set. CI now tests Python 3.12 and 3.13.
- Bumped GitHub Actions to their Node 24 runtimes (`checkout@v5`,
  `setup-uv@v7`, `upload-artifact@v7`, `download-artifact@v8`).

## [0.3.0] - 2026-05-31

### Added

- **PyPI packaging**: published as `quality-triage` (the CLI command remains
  `code-review`; the import package remains `code_review_agent`).
- **Trusted Publishing (OIDC)** release workflow (`.github/workflows/release.yml`)
  triggered on `v*` tags, with a tag↔version guard and least-privilege
  permissions. No API token stored in CI.
- **CI** split into three jobs: lint + format check, test matrix
  (Python 3.10 / 3.11 / 3.12), and build + `twine check` + artifact upload.
- **Ruff lint config** (`[tool.ruff.lint]`) with rule set `E, W, F, I, B, UP,
  C4, SIM` enforced in CI.
- Package metadata: authors, keywords, trove classifiers, and `[project.urls]`.
- Maintenance/release runbook and PyPI install docs in `README.md`; dev/CI/
  release section in `AGENTS.md`.

### Changed

- Detector dependencies now carry `>=` lower bounds matching their PyPI
  releases (`ml-code-smell-detector>=0.1.2`, `code-quality-analyzer>=0.2.2`,
  `tdsuite>=0.1.2`); `[tool.uv.sources]` still pins dev installs to GitHub `main`.
- Report footer now derives the version from `__version__` instead of a
  hardcoded string.
- Branding strings (`--version` lookup, install hints, MCP/web server titles,
  `doctor` header) updated to the `quality-triage` / `code-review` names.

### Fixed

- `code-review --version` no longer crashes: the version lookup now references
  the correct distribution name (`quality-triage`).

## [0.2.0]

- Integrated all three detector suites (ML smells, Python smells, technical-debt
  classification) with selection/fix controls, a FastAPI web UI, and full
  smell-catalog coverage. Native ONNX ensemble engine for TD classification.

## [0.1.0]

- Initial release: AI-powered code review agent.

[0.3.0]: https://github.com/KarthikShivasankar/quality-triage/releases/tag/v0.3.0
