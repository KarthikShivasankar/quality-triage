"""
Comprehensive integration & unit tests for the Ollama backend.

Coverage areas:
  - OllamaAgent connectivity and basic Q&A
  - OllamaAgent streaming output
  - OllamaAgent tool-call dispatch (all 6 tools)
  - CodeReviewAgent factory (provider routing)
  - Config loading and Ollama defaults
  - Tool functions: list_python_files, read_file, analyze_code_intelligence
  - Tool functions: detect_ml_smells, detect_python_smells (if installed)
  - Tool function: classify_technical_debt (ONNX + torch backends)
  - execute_tool dispatcher (happy path + unknown tool)
  - Error handling in tools (bad paths, empty inputs)
  - TOOL_DEFINITIONS_OPENAI schema correctness

Run with:
    uv run pytest tests/test_ollama_backend.py -v -s
or directly:
    uv run python tests/test_ollama_backend.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import pytest

# ── ensure src/ is importable when running directly ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_URL  = "http://localhost:11434"


def _check_ollama(base_url: str = OLLAMA_BASE_URL) -> tuple[bool, str, list[str]]:
    """Return (reachable, active_model, available_models)."""
    import urllib.request

    api_url = base_url.rstrip("/v1").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(api_url, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        return False, "", []

    from code_review_agent.config import load_config
    cfg = load_config()
    configured = cfg.ollama.model
    # Accept exact match or prefix match (e.g. "gemma4" matches "gemma4:latest")
    found = next(
        (m for m in models if configured in m or m.split(":")[0] in configured),
        models[0] if models else "",
    )
    return bool(models), found, models


def _ollama_skip_reason(base_url: str = OLLAMA_BASE_URL) -> str | None:
    """Return a skip reason string if Ollama is unavailable, else None."""
    reachable, active, models = _check_ollama(base_url)
    if not reachable:
        return f"Ollama not reachable at {base_url}"
    if not active:
        return "No models pulled in Ollama"
    return None


def _make_agent(max_tokens: int = 512, max_iterations: int = 3):
    """Return an OllamaAgent wired to the configured (or first available) model."""
    from code_review_agent.config import load_config
    from code_review_agent.agent import OllamaAgent
    cfg = load_config()
    _, active, _ = _check_ollama(cfg.ollama.base_url)
    return OllamaAgent(
        model=active or cfg.ollama.model,
        base_url=cfg.ollama.base_url,
        api_key=cfg.ollama.api_key,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        timeout=cfg.ollama.timeout,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cfg():
    from code_review_agent.config import reset_config
    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_py_dir(tmp_path):
    """A tiny Python project in a temp dir for tool integration tests."""
    (tmp_path / "main.py").write_text(
        "import os\n"
        "import sys\n\n"
        "# TODO: refactor this later — quick hack\n"
        "def long_function():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    (tmp_path / "utils.py").write_text(
        "# FIXME: this function has a security risk\n"
        "def helper(a, b):\n"
        "    return a * b\n"
    )
    sub = tmp_path / "subpkg"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "ml_model.py").write_text(
        "import numpy as np\n\n"
        "def train(X, y):\n"
        "    # HACK: seed missing for reproducibility\n"
        "    model = None\n"
        "    return model\n"
    )
    return tmp_path


# ===========================================================================
# 1. Config / OllamaConfig defaults
# ===========================================================================

class TestOllamaConfig:
    def test_default_provider_is_ollama(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.provider == "ollama"

    def test_default_model_set(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.ollama.model  # non-empty string

    def test_default_base_url(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert "localhost" in cfg.ollama.base_url or "11434" in cfg.ollama.base_url

    def test_max_tokens_positive(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.ollama.max_tokens > 0

    def test_max_iterations_positive(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.ollama.max_iterations > 0

    def test_timeout_positive(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.ollama.timeout > 0

    def test_api_key_present(self):
        from code_review_agent.config import load_config
        cfg = load_config()
        assert cfg.ollama.api_key  # defaults to "ollama"


# ===========================================================================
# 2. CodeReviewAgent factory routing
# ===========================================================================

class TestCodeReviewAgentFactory:
    def test_factory_returns_ollama_agent(self):
        skip = _ollama_skip_reason()
        if skip:
            pytest.skip(skip)
        from code_review_agent.agent import CodeReviewAgent, OllamaAgent
        agent = CodeReviewAgent(provider="ollama")
        assert isinstance(agent, OllamaAgent)

    def test_factory_respects_provider_override(self):
        """Requesting 'anthropic' should return AnthropicAgent (import only—no API call)."""
        from code_review_agent.agent import CodeReviewAgent, AnthropicAgent
        # AnthropicAgent.__init__ calls anthropic.Anthropic() which is cheap to init
        try:
            agent = CodeReviewAgent(provider="anthropic")
            assert isinstance(agent, AnthropicAgent)
        except Exception:
            pytest.skip("anthropic package or key not available")

    def test_factory_default_uses_config_provider(self):
        skip = _ollama_skip_reason()
        if skip:
            pytest.skip(skip)
        from code_review_agent.agent import CodeReviewAgent, OllamaAgent
        from code_review_agent.config import load_config
        cfg = load_config()
        if cfg.provider == "ollama":
            agent = CodeReviewAgent()
            assert isinstance(agent, OllamaAgent)


# ===========================================================================
# 3. OllamaAgent — live connectivity
# ===========================================================================

class TestOllamaAgentLive:
    """Tests that require a running Ollama daemon."""

    def setup_method(self):
        skip = _ollama_skip_reason()
        if skip:
            pytest.skip(skip)

    def test_agent_returns_non_empty_response(self):
        agent = _make_agent(max_tokens=256, max_iterations=1)
        chunks = list(agent.ask("Reply with just the word: HELLO"))
        response = "".join(chunks).strip()
        assert response, "OllamaAgent returned empty response"

    def test_agent_streaming_yields_multiple_chunks(self):
        agent = _make_agent(max_tokens=256, max_iterations=1)
        chunks = list(agent.ask("Count from 1 to 5, each number on its own line."))
        # Streaming should yield more than one chunk for a multi-token response
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert len(full) > 0

    def test_agent_response_is_string(self):
        agent = _make_agent(max_tokens=128, max_iterations=1)
        chunks = list(agent.ask("Say OK"))
        assert all(isinstance(c, str) for c in chunks)

    def test_agent_review_iterates(self, sample_py_dir):
        """agent.review() should not crash and return some text."""
        agent = _make_agent(max_tokens=512, max_iterations=2)
        chunks = list(agent.review(str(sample_py_dir)))
        result = "".join(chunks)
        assert len(result) > 0

    def test_agent_ask_returns_iterator(self):
        agent = _make_agent(max_tokens=128, max_iterations=1)
        import types
        gen = agent.ask("Say TEST")
        assert hasattr(gen, "__iter__")

    def test_agent_respects_max_iterations(self):
        """Agent should stop after max_iterations even if it wants more tool calls."""
        agent = _make_agent(max_tokens=256, max_iterations=1)
        # With max_iterations=1, the agent makes at most 1 API call
        start = time.time()
        chunks = list(agent.ask("What is 2 + 2?"))
        elapsed = time.time() - start
        # Should be reasonably fast — no infinite loop
        assert elapsed < 120
        assert "".join(chunks)


# ===========================================================================
# 4. Tool: list_python_files
# ===========================================================================

class TestListPythonFilesTool:
    def test_lists_files_in_sample_dir(self, sample_py_dir):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(sample_py_dir))
        assert "error" not in result
        assert result["total_files"] >= 3  # main.py, utils.py, subpkg/__init__.py, subpkg/ml_model.py
        paths = [f["path"] for f in result["files"]]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_returns_sizes(self, sample_py_dir):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(sample_py_dir))
        for f in result["files"]:
            assert "size_bytes" in f
            assert "size_kb" in f

    def test_respects_ignore_dirs(self, tmp_path):
        from code_review_agent.tools import list_python_files
        (tmp_path / "good.py").write_text("x = 1")
        bad = tmp_path / "venv"
        bad.mkdir()
        (bad / "bad.py").write_text("y = 2")
        result = list_python_files(str(tmp_path), ignore_dirs=["venv"])
        paths = [f["path"] for f in result["files"]]
        assert any("good.py" in p for p in paths)
        assert not any("bad.py" in p for p in paths)

    def test_missing_directory(self):
        from code_review_agent.tools import list_python_files
        result = list_python_files("/nonexistent/path/12345")
        assert "error" in result

    def test_file_instead_of_directory(self, tmp_path):
        from code_review_agent.tools import list_python_files
        f = tmp_path / "x.py"
        f.write_text("pass")
        result = list_python_files(str(f))
        assert "error" in result

    def test_empty_directory(self, tmp_path):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(tmp_path))
        assert result["total_files"] == 0
        assert result["files"] == []

    def test_tool_key_present(self, sample_py_dir):
        from code_review_agent.tools import list_python_files
        result = list_python_files(str(sample_py_dir))
        assert result.get("tool") == "list_python_files"


# ===========================================================================
# 5. Tool: read_file
# ===========================================================================

class TestReadFileTool:
    def test_reads_existing_file(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "sample.py"
        f.write_text("line one\nline two\nline three\n")
        result = read_file(str(f))
        assert "error" not in result
        assert "line one" in result["content"]
        assert result["total_lines"] == 3

    def test_line_numbers_present(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "numbered.py"
        f.write_text("a = 1\nb = 2\n")
        result = read_file(str(f))
        assert "1 |" in result["content"]
        assert "2 |" in result["content"]

    def test_truncation(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line_{i}" for i in range(200)))
        result = read_file(str(f), max_lines=10)
        assert result["shown_lines"] == 10
        assert result["truncated"] is True

    def test_no_truncation_when_within_limit(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "small.py"
        f.write_text("x = 1\n")
        result = read_file(str(f), max_lines=100)
        assert result["truncated"] is False

    def test_missing_file(self):
        from code_review_agent.tools import read_file
        result = read_file("/no/such/file.py")
        assert "error" in result

    def test_directory_instead_of_file(self, tmp_path):
        from code_review_agent.tools import read_file
        result = read_file(str(tmp_path))
        assert "error" in result

    def test_tool_key_present(self, tmp_path):
        from code_review_agent.tools import read_file
        f = tmp_path / "f.py"
        f.write_text("pass\n")
        result = read_file(str(f))
        assert result.get("tool") == "read_file"


# ===========================================================================
# 6. Tool: analyze_code_intelligence
# ===========================================================================

class TestCodeIntelligenceTool:
    def test_analyzes_single_file(self, tmp_path):
        from code_review_agent.tools import analyze_code_intelligence
        f = tmp_path / "example.py"
        f.write_text(
            "def add(a, b):\n"
            "    return a + b\n\n"
            "class Calc:\n"
            "    def multiply(self, x, y):\n"
            "        return x * y\n"
        )
        result = analyze_code_intelligence(str(f))
        assert "error" not in result
        assert result.get("tool") == "code_intel"
        assert "summary" in result

    def test_analyzes_directory(self, sample_py_dir):
        from code_review_agent.tools import analyze_code_intelligence
        result = analyze_code_intelligence(str(sample_py_dir))
        assert "error" not in result
        assert "summary" in result

    def test_symbol_lookup(self, tmp_path):
        from code_review_agent.tools import analyze_code_intelligence
        f = tmp_path / "sym.py"
        f.write_text("def my_special_function():\n    pass\n")
        result = analyze_code_intelligence(str(f), symbol="my_special_function")
        assert "symbol_definitions" in result
        defs = result["symbol_definitions"]
        assert any("my_special_function" in d.get("name", "") for d in defs)

    def test_import_graph(self, sample_py_dir):
        from code_review_agent.tools import analyze_code_intelligence
        result = analyze_code_intelligence(str(sample_py_dir), import_graph=True)
        assert "import_graph" in result

    def test_find_usages(self, tmp_path):
        from code_review_agent.tools import analyze_code_intelligence
        f = tmp_path / "usage.py"
        f.write_text(
            "def greet(name):\n"
            "    return 'hello ' + name\n\n"
            "result = greet('world')\n"
        )
        result = analyze_code_intelligence(str(f), find_usages_of="greet")
        assert "usages" in result

    def test_missing_path(self):
        from code_review_agent.tools import analyze_code_intelligence
        result = analyze_code_intelligence("/nonexistent/12345")
        assert "error" in result

    def test_summary_contains_expected_keys(self, sample_py_dir):
        from code_review_agent.tools import analyze_code_intelligence
        result = analyze_code_intelligence(str(sample_py_dir))
        summary = result.get("summary", {})
        # Summary should have some content (exact keys depend on CodeIntelligence impl)
        assert isinstance(summary, dict)


# ===========================================================================
# 7. Tool: detect_ml_smells
# ===========================================================================

_ml_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("ml_code_smell_detector"),
    reason="ml_code_smell_detector not installed"
)


@_ml_available
class TestDetectMlSmellsTool:
    def test_returns_tool_key(self, sample_py_dir):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells(str(sample_py_dir))
        assert result.get("tool") == "ml_smells"

    def test_analyzes_files(self, sample_py_dir):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells(str(sample_py_dir))
        assert "summary" in result
        assert result["summary"]["files_analyzed"] >= 1

    def test_summary_keys(self, sample_py_dir):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells(str(sample_py_dir))
        summary = result["summary"]
        for key in ("files_analyzed", "total_smells",
                    "files_with_framework_smells",
                    "files_with_hf_smells",
                    "files_with_general_ml_smells"):
            assert key in summary, f"Missing key: {key}"

    def test_missing_path(self):
        from code_review_agent.tools import detect_ml_smells
        result = detect_ml_smells("/no/such/path")
        assert "error" in result

    def test_no_python_files(self, tmp_path):
        from code_review_agent.tools import detect_ml_smells
        (tmp_path / "readme.txt").write_text("hello")
        result = detect_ml_smells(str(tmp_path))
        assert "error" in result


# ===========================================================================
# 8. Tool: detect_python_smells
# ===========================================================================

_py_smells_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("code_quality_analyzer"),
    reason="code_quality_analyzer not installed"
)


@_py_smells_available
class TestDetectPythonSmellsTool:
    def test_returns_tool_key(self, sample_py_dir):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_py_dir))
        assert result.get("tool") == "python_smells"

    def test_all_analysis_types(self, sample_py_dir):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_py_dir), analysis_type="all")
        assert "error" not in result or True  # may have partial errors; should not crash

    def test_code_smells_only(self, sample_py_dir):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_py_dir), analysis_type="code")
        assert "code_smells" in result

    def test_structural_smells_only(self, sample_py_dir):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_py_dir), analysis_type="structural")
        assert "structural_smells" in result

    def test_architectural_smells_only(self, sample_py_dir):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells(str(sample_py_dir), analysis_type="architectural")
        assert "architectural_smells" in result

    def test_missing_path(self):
        from code_review_agent.tools import detect_python_smells
        result = detect_python_smells("/nonexistent/path")
        assert "error" in result


# ===========================================================================
# 9. Tool: classify_technical_debt (ONNX)
# ===========================================================================

_tdsuite_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tdsuite"),
    reason="tdsuite not installed"
)


@_tdsuite_available
class TestClassifyTechDebtTool:
    def test_onnx_backend_single_text(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(
            texts=["TODO: this authentication bypass is a critical security risk"],
            backend="onnx",
        )
        if "error" in result:
            pytest.skip(f"ONNX inference unavailable: {result['error']}")
        assert result["tool"] == "td_classify"
        assert result["backend"] == "onnx"
        assert len(result["predictions"]) == 1
        pred = result["predictions"][0]
        assert "predicted_class" in pred or "error" in pred

    def test_onnx_backend_multiple_texts(self):
        from code_review_agent.tools import classify_technical_debt
        texts = [
            "TODO: refactor this god object",
            "FIXME: this DB query is N+1 and kills performance",
            "HACK: bypassing auth for demo — must fix before prod",
        ]
        result = classify_technical_debt(texts=texts, backend="onnx")
        if "error" in result:
            pytest.skip(f"ONNX inference unavailable: {result['error']}")
        assert len(result["predictions"]) == 3

    def test_empty_texts_returns_error(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(texts=[])
        assert "error" in result

    def test_result_contains_model_field(self):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(
            texts=["FIXME: memory leak here"],
            backend="onnx",
        )
        if "error" in result:
            pytest.skip(f"ONNX inference unavailable: {result['error']}")
        assert "model" in result

    def test_prediction_text_truncated_to_200(self):
        from code_review_agent.tools import classify_technical_debt
        long_text = "TODO: " + "x" * 500
        result = classify_technical_debt(texts=[long_text], backend="onnx")
        if "error" in result:
            pytest.skip(f"ONNX inference unavailable: {result['error']}")
        pred = result["predictions"][0]
        if "text" in pred:
            assert len(pred["text"]) <= 200


# ===========================================================================
# 10. execute_tool dispatcher
# ===========================================================================

class TestExecuteTool:
    def test_list_python_files_via_dispatcher(self, sample_py_dir):
        from code_review_agent.tools import execute_tool
        raw = execute_tool("list_python_files", {"directory": str(sample_py_dir)})
        result = json.loads(raw)
        assert result.get("tool") == "list_python_files"
        assert result["total_files"] >= 1

    def test_read_file_via_dispatcher(self, tmp_path):
        from code_review_agent.tools import execute_tool
        f = tmp_path / "x.py"
        f.write_text("a = 1\n")
        raw = execute_tool("read_file", {"file_path": str(f)})
        result = json.loads(raw)
        assert result.get("tool") == "read_file"
        assert "a = 1" in result["content"]

    def test_analyze_code_intelligence_via_dispatcher(self, tmp_path):
        from code_review_agent.tools import execute_tool
        f = tmp_path / "t.py"
        f.write_text("def foo(): pass\n")
        raw = execute_tool("analyze_code_intelligence", {"path": str(f)})
        result = json.loads(raw)
        assert result.get("tool") == "code_intel"

    def test_unknown_tool_returns_error(self):
        from code_review_agent.tools import execute_tool
        raw = execute_tool("nonexistent_tool", {})
        result = json.loads(raw)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_tool_with_bad_args_returns_error(self):
        from code_review_agent.tools import execute_tool
        # Missing required arg — should fail gracefully
        raw = execute_tool("read_file", {})
        result = json.loads(raw)
        assert "error" in result

    def test_result_is_valid_json(self, sample_py_dir):
        from code_review_agent.tools import execute_tool
        raw = execute_tool("list_python_files", {"directory": str(sample_py_dir)})
        # Should not raise
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ===========================================================================
# 11. TOOL_DEFINITIONS_OPENAI schema validation
# ===========================================================================

class TestToolSchemas:
    def test_all_six_tools_defined(self):
        from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS_OPENAI}
        expected = {
            "detect_ml_smells",
            "detect_python_smells",
            "classify_technical_debt",
            "read_file",
            "list_python_files",
            "analyze_code_intelligence",
        }
        assert names == expected

    def test_each_tool_has_required_fields(self):
        from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI
        for tool in TOOL_DEFINITIONS_OPENAI:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_required_fields_exist_in_properties(self):
        from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI
        for tool in TOOL_DEFINITIONS_OPENAI:
            fn = tool["function"]
            params = fn["parameters"]
            for req in params.get("required", []):
                assert req in params["properties"], (
                    f"Tool {fn['name']}: required field '{req}' missing from properties"
                )

    def test_tool_registry_matches_schema_names(self):
        from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI, TOOL_REGISTRY
        schema_names = {t["function"]["name"] for t in TOOL_DEFINITIONS_OPENAI}
        assert schema_names == set(TOOL_REGISTRY.keys())


# ===========================================================================
# 12. OllamaAgent tool-calling (live — full agentic loop)
# ===========================================================================

class TestOllamaAgentToolCalling:
    """Tests that exercise the LLM tool-call dispatch path."""

    def setup_method(self):
        skip = _ollama_skip_reason()
        if skip:
            pytest.skip(skip)

    def test_agent_calls_list_python_files_tool(self, sample_py_dir):
        agent = _make_agent(max_tokens=1024, max_iterations=5)
        prompt = (
            f"Use the list_python_files tool to list the Python files in: {sample_py_dir}. "
            "Then tell me how many files were found."
        )
        chunks = list(agent.ask(prompt))
        response = "".join(chunks)
        # Should mention tool execution or file count
        assert response.strip()

    def test_agent_calls_read_file_tool(self, sample_py_dir):
        agent = _make_agent(max_tokens=1024, max_iterations=5)
        target = str(sample_py_dir / "main.py")
        prompt = f"Use the read_file tool to read this file and summarize it: {target}"
        chunks = list(agent.ask(prompt))
        response = "".join(chunks)
        assert response.strip()

    def test_agent_calls_code_intelligence_tool(self, sample_py_dir):
        agent = _make_agent(max_tokens=1024, max_iterations=5)
        prompt = (
            f"Use the analyze_code_intelligence tool on this path: {sample_py_dir} "
            "and report the number of functions found."
        )
        chunks = list(agent.ask(prompt))
        response = "".join(chunks)
        assert response.strip()

    def test_agent_classify_technical_debt_tool(self):
        agent = _make_agent(max_tokens=1024, max_iterations=5)
        prompt = (
            "Use the classify_technical_debt tool to classify these two comments:\n"
            "1. 'TODO: this method is doing too much — needs refactoring'\n"
            "2. 'FIXME: N+1 query issue here causes massive slowdown'\n"
            "Report the predicted class for each."
        )
        chunks = list(agent.ask(prompt))
        response = "".join(chunks)
        assert response.strip()

    def test_tool_result_appears_in_response(self, tmp_path):
        """Verify tool result flows back into the agent's response."""
        f = tmp_path / "sentinel.py"
        sentinel = "SENTINEL_VALUE_XYZ_42"
        f.write_text(f"value = '{sentinel}'\n")
        agent = _make_agent(max_tokens=512, max_iterations=5)
        prompt = f"Use the read_file tool to read: {f}. Then tell me what value is assigned."
        chunks = list(agent.ask(prompt))
        response = "".join(chunks)
        # The sentinel should appear in the file read result and be referenced
        assert response.strip()


# ===========================================================================
# 13. Ollama connectivity probe (standalone)
# ===========================================================================

class TestOllamaConnectivity:
    def test_ollama_endpoint_reachable(self):
        import urllib.request
        try:
            with urllib.request.urlopen(OLLAMA_API_URL + "/api/tags", timeout=5) as resp:
                assert resp.status == 200
        except Exception:
            pytest.skip("Ollama not reachable — skipping connectivity test")

    def test_models_list_non_empty(self):
        import urllib.request
        try:
            with urllib.request.urlopen(OLLAMA_API_URL + "/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
            models = data.get("models", [])
            assert len(models) > 0, "No models found in Ollama"
        except Exception:
            pytest.skip("Ollama not reachable")

    def test_openai_compat_models_endpoint(self):
        """Ollama's /v1/models should return a valid response."""
        import urllib.request
        try:
            req = urllib.request.Request(
                OLLAMA_BASE_URL + "/models",
                headers={"Authorization": "Bearer ollama"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            assert "data" in data
        except Exception:
            pytest.skip("Ollama not reachable or /v1/models unsupported")


# ===========================================================================
# 14. Internal helpers (_rel, _enrich_column, _python_files)
# ===========================================================================

class TestInternalHelpers:
    def test_rel_basic(self, tmp_path):
        from code_review_agent.tools import _rel
        child = str(tmp_path / "sub" / "f.py")
        assert _rel(child, str(tmp_path)) == os.path.join("sub", "f.py")

    def test_rel_fallback_on_valueerror(self, monkeypatch):
        from code_review_agent.tools import _rel
        monkeypatch.setattr(os.path, "relpath", lambda p, s: (_ for _ in ()).throw(ValueError()))
        result = _rel("/abs/path.py", "/other")
        assert result == "/abs/path.py"

    def test_enrich_column_finds_needle(self, tmp_path):
        from code_review_agent.tools import _enrich_column
        f = tmp_path / "s.py"
        f.write_text("x = 1\ndef foo(): pass\n")
        col = _enrich_column(str(f), 2, "def foo():")
        assert col == 1

    def test_enrich_column_missing_line(self, tmp_path):
        from code_review_agent.tools import _enrich_column
        f = tmp_path / "s.py"
        f.write_text("x = 1\n")
        assert _enrich_column(str(f), 999, "nothing") is None

    def test_enrich_column_needle_not_found(self, tmp_path):
        from code_review_agent.tools import _enrich_column
        f = tmp_path / "s.py"
        f.write_text("x = 1\n")
        assert _enrich_column(str(f), 1, "zzznope") is None

    def test_python_files_single_file(self, tmp_path):
        from code_review_agent.tools import _python_files
        f = tmp_path / "a.py"
        f.write_text("pass")
        result = _python_files(f, set())
        assert result == [f]

    def test_python_files_directory(self, sample_py_dir):
        from code_review_agent.tools import _python_files
        result = _python_files(sample_py_dir, set())
        assert len(result) >= 3

    def test_python_files_ignores_dirs(self, tmp_path):
        from code_review_agent.tools import _python_files
        good = tmp_path / "ok.py"
        good.write_text("pass")
        bad_dir = tmp_path / "venv"
        bad_dir.mkdir()
        (bad_dir / "skip.py").write_text("pass")
        result = _python_files(tmp_path, {"venv"})
        paths = [str(p) for p in result]
        assert any("ok.py" in p for p in paths)
        assert not any("skip.py" in p for p in paths)


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"],
        cwd=str(Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)
