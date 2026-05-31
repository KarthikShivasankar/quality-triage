"""
Universal MCP (Model Context Protocol) server for code-review-agent.

Exposes the analysis tools over stdio so ANY MCP-capable harness
(Claude Code, Codex, Cursor, Pi, Antigravity, …) can call them:

    detect_ml_smells, detect_python_smells, classify_technical_debt,
    classify_technical_debt_all, classify_technical_debt_ensemble,
    classify_github_issues, list_td_categories,
    analyze_code_intelligence, list_python_files, read_file

The ``mcp`` SDK is imported lazily so importing this module never fails when
the optional extra is not installed. Install with::

    uv sync --extra mcp        # or:  pip install "code-review-agent[mcp]"

Run with::

    code-review-mcp            # console script
    uv run python -m code_review_agent.mcp_server
"""

from __future__ import annotations

import json
from typing import Any

from code_review_agent import tools as _tools

_INSTALL_HINT = (
    "The 'mcp' package is required to run the MCP server.\n"
    "Install it with:  uv sync --extra mcp   (or  pip install \"code-review-agent[mcp]\")"
)


def _result_text(payload: Any) -> str:
    """Serialise a tool result to a compact JSON string."""
    return json.dumps(payload, default=str, indent=2)


def build_server():
    """Create and return a configured FastMCP server instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_INSTALL_HINT) from exc

    server = FastMCP("code-review-agent")

    @server.tool()
    def detect_ml_smells(path: str, ignore_dirs: list[str] | None = None) -> str:
        """Detect ML-specific anti-patterns (pandas/numpy/sklearn/torch/TF/HF)."""
        return _result_text(_tools.detect_ml_smells(path, ignore_dirs=ignore_dirs))

    @server.tool()
    def detect_python_smells(
        path: str,
        analysis_type: str = "all",
        ignore_dirs: list[str] | None = None,
    ) -> str:
        """Detect code / architectural / structural Python smells."""
        return _result_text(
            _tools.detect_python_smells(path, analysis_type=analysis_type, ignore_dirs=ignore_dirs)
        )

    @server.tool()
    def classify_technical_debt(
        texts: list[str],
        category: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
    ) -> str:
        """Binary per-category technical-debt classification of text snippets."""
        return _result_text(
            _tools.classify_technical_debt(
                texts, category=category, model_path=model_path, device=device
            )
        )

    @server.tool()
    def analyze_code_intelligence(
        path: str,
        symbol: str | None = None,
        find_usages_of: str | None = None,
        metrics_only: bool = False,
        import_graph: bool = False,
        ignore_dirs: list[str] | None = None,
        top_n: int | None = None,
    ) -> str:
        """AST code intelligence: symbols, metrics, import graph, usages."""
        return _result_text(
            _tools.analyze_code_intelligence(
                path,
                symbol=symbol,
                find_usages_of=find_usages_of,
                metrics_only=metrics_only,
                import_graph=import_graph,
                ignore_dirs=ignore_dirs,
                top_n=top_n,
            )
        )

    @server.tool()
    def classify_technical_debt_all(
        texts: list[str],
        categories: list[str] | None = None,
        device: str | None = None,
    ) -> str:
        """Classify snippets against every (or chosen) TD category model (multi-label)."""
        return _result_text(
            _tools.classify_technical_debt_all(texts, categories=categories, device=device)
        )

    @server.tool()
    def classify_technical_debt_ensemble(
        texts: list[str],
        model_names: list[str] | None = None,
        categories: list[str] | None = None,
        weights: list[float] | None = None,
        device: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Classify snippets with a weighted ensemble of TD models.

        Runs on the native torch-free ONNX ensemble by default; pass
        backend="torch" to force the PyTorch engine.
        """
        return _result_text(
            _tools.classify_technical_debt_ensemble(
                texts, model_names=model_names, categories=categories,
                weights=weights, device=device, backend=backend,
            )
        )

    @server.tool()
    def classify_github_issues(
        repo: str,
        category: str | None = None,
        state: str = "all",
        limit: int = 50,
        device: str | None = None,
    ) -> str:
        """Fetch GitHub issues (owner/repo) and classify them for technical debt."""
        return _result_text(
            _tools.classify_github_issues(
                repo, category=category, state=state, limit=limit, device=device
            )
        )

    @server.tool()
    def list_td_categories() -> str:
        """List available technical-debt categories and their HF model ids."""
        return _result_text(_tools.list_td_categories())

    @server.tool()
    def list_python_files(directory: str, ignore_dirs: list[str] | None = None) -> str:
        """List all Python files in a project directory with sizes."""
        return _result_text(_tools.list_python_files(directory, ignore_dirs=ignore_dirs))

    @server.tool()
    def read_file(file_path: str, max_lines: int | None = None) -> str:
        """Read a Python file with line numbers."""
        return _result_text(_tools.read_file(file_path, max_lines=max_lines))

    return server


def main() -> None:
    """Console-script entry point: run the stdio MCP server."""
    try:
        server = build_server()
    except ImportError as exc:
        import sys

        print(str(exc), file=sys.stderr)
        sys.exit(1)
    server.run()


if __name__ == "__main__":
    main()
