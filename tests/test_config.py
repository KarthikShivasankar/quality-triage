"""Tests for code_review_agent.config"""

from code_review_agent.config import (
    DEFAULT_LITELLM_MODEL,
    DEFAULT_OLLAMA_NATIVE_MODEL,
    ToolsConfig,
    get_config,
    get_thresholds,
    load_config,
    reset_config,
)


def test_load_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.llm.model == DEFAULT_LITELLM_MODEL
    assert cfg.provider == "ollama"
    assert cfg.aliases["local"] == DEFAULT_LITELLM_MODEL
    assert cfg.aliases["test"] == DEFAULT_LITELLM_MODEL
    assert DEFAULT_OLLAMA_NATIVE_MODEL in cfg.llm.model
    assert cfg._source == "defaults"


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "llm:\n  model: groq/llama-3.3-70b-versatile\n  timeout: 9\n"
    )
    cfg = load_config()
    assert cfg.llm.model == "groq/llama-3.3-70b-versatile"
    assert cfg.llm.timeout == 9
    assert cfg.llm.max_tokens == 8192


def test_load_config_explicit_path(tmp_path):
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("llm:\n  model: openai/gpt-4.1\n")
    cfg = load_config(str(config_file))
    assert cfg.llm.model == "openai/gpt-4.1"
    assert cfg._source == str(config_file)


def test_legacy_hf_gguf_ollama_block_migrates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "provider: ollama\n"
        "ollama:\n"
        "  model: hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M\n"
        "  base_url: http://localhost:11434\n"
    )
    cfg = load_config()
    assert cfg.llm.model == "ollama/hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"


def test_legacy_ollama_block_migrates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "provider: ollama\n"
        "ollama:\n"
        "  model: gemma4:latest\n"
        "  base_url: http://localhost:11434/v1\n"
        "  max_tokens: 1024\n"
    )
    cfg = load_config()
    assert cfg.llm.model == "ollama/gemma4:latest"
    assert cfg.llm.api_base == "http://localhost:11434"
    assert cfg.llm.max_tokens == 1024


def test_legacy_anthropic_block_migrates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "provider: anthropic\n"
        "anthropic:\n"
        "  model: claude-opus-4-6\n"
        "  max_iterations: 7\n"
    )
    cfg = load_config()
    assert cfg.llm.model == "anthropic/claude-opus-4-6"
    assert cfg.llm.max_iterations == 7
    assert cfg.provider == "anthropic"


def test_load_config_unknown_keys_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  model: ollama/x\nunknown_key: foo\n")
    cfg = load_config()
    assert cfg.llm.model == "ollama/x"


def test_aliases_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("aliases:\n  lab: openai/my-lab\n")
    cfg = load_config()
    assert cfg.aliases["lab"] == "openai/my-lab"
    assert "frontier" in cfg.aliases


def test_tools_config_defaults():
    tc = ToolsConfig()
    assert ".git" in tc.ignore_dirs
    assert "__pycache__" in tc.ignore_dirs
    assert tc.read_file_max_lines == 500


def test_get_config_singleton(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_config()
    assert get_config() is get_config()


def test_reset_config_forces_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_config()
    cfg1 = get_config()
    reset_config()
    cfg2 = get_config()
    assert cfg1 is not cfg2


def test_get_config_with_path(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("llm:\n  model: openai/gpt-4.1\n")
    reset_config()
    cfg = get_config(str(config_file))
    assert cfg.llm.model == "openai/gpt-4.1"
    reset_config()


def test_get_thresholds_missing_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    result = get_thresholds(cfg, "code_smells")
    assert isinstance(result, dict)


def test_get_thresholds_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "code_smells:\n  LONG_METHOD:\n    value: 40\n"
    )
    cfg = load_config()
    result = get_thresholds(cfg, "code_smells")
    assert result == {"LONG_METHOD": 40}


def test_report_fail_on_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.report.fail_on == "none"
    assert cfg.report.default_format == "markdown"


def test_github_issue_fetch_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.github.fetch_issues is True
    assert cfg.github.issue_limit == 10
    (tmp_path / "config.yaml").write_text(
        "github:\n  fetch_issues: false\n  issue_limit: 30\n"
    )
    cfg = load_config()
    assert cfg.github.fetch_issues is False
    assert cfg.github.issue_limit == 30
