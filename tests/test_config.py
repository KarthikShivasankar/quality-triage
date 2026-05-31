"""
Tests for code_review_agent.config
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# load_config / defaults
# ---------------------------------------------------------------------------

def test_load_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    """load_config with no config.yaml anywhere should return defaults."""
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.provider == "ollama"
    assert cfg.ollama.model == "qwen3.5:4b"
    assert cfg.anthropic.model == "claude-opus-4-6"
    assert cfg._source == "defaults"


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    """load_config should parse a config.yaml in CWD."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: anthropic\nollama:\n  model: mistral\n")
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.provider == "anthropic"
    assert cfg.ollama.model == "mistral"
    # Unset keys should still use defaults
    assert cfg.anthropic.model == "claude-opus-4-6"


def test_load_config_explicit_path(tmp_path):
    """load_config with an explicit path argument should use that file."""
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("provider: anthropic\n")
    from code_review_agent.config import load_config
    cfg = load_config(str(config_file))
    assert cfg.provider == "anthropic"
    assert cfg._source == str(config_file)


def test_load_config_unknown_keys_ignored(tmp_path, monkeypatch):
    """Unknown top-level YAML keys should not raise errors."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: ollama\nunknown_key: foo\n")
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.provider == "ollama"


def test_tools_config_defaults():
    """ToolsConfig should carry sensible ignore_dirs by default."""
    from code_review_agent.config import ToolsConfig
    tc = ToolsConfig()
    assert ".git" in tc.ignore_dirs
    assert "__pycache__" in tc.ignore_dirs
    assert tc.read_file_max_lines == 500


# ---------------------------------------------------------------------------
# Singleton (get_config / reset_config)
# ---------------------------------------------------------------------------

def test_get_config_singleton(tmp_path, monkeypatch):
    """get_config() should return the same object on repeated calls."""
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import get_config, reset_config
    reset_config()
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_reset_config_forces_reload(tmp_path, monkeypatch):
    """reset_config() should force a fresh load on next get_config()."""
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import get_config, reset_config
    reset_config()
    cfg1 = get_config()
    reset_config()
    cfg2 = get_config()
    # Both are fresh objects (not the same instance after reset)
    assert cfg1 is not cfg2


def test_get_config_with_path(tmp_path):
    """get_config(path=...) should load the specified file."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("provider: anthropic\n")
    from code_review_agent.config import get_config, reset_config
    reset_config()
    cfg = get_config(str(config_file))
    assert cfg.provider == "anthropic"
    reset_config()


# ---------------------------------------------------------------------------
# get_thresholds
# ---------------------------------------------------------------------------

def test_get_thresholds_missing_key(tmp_path, monkeypatch):
    """get_thresholds returns empty dict when key absent."""
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import load_config, get_thresholds
    cfg = load_config()
    result = get_thresholds(cfg, "code_smells")
    assert isinstance(result, dict)


def test_get_thresholds_present(tmp_path, monkeypatch):
    """get_thresholds returns the nested dict when present in config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "code_smells:\n  LONG_METHOD:\n    value: 40\n"
    )
    from code_review_agent.config import load_config, get_thresholds
    cfg = load_config()
    result = get_thresholds(cfg, "code_smells")
    assert result == {"LONG_METHOD": {"value": 40}}


# ---------------------------------------------------------------------------
# OpenAI provider config + provider defaults
# ---------------------------------------------------------------------------

def test_openai_config_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.openai.model == "gpt-4o-mini"
    assert cfg.openai.base_url == "https://api.openai.com/v1"
    assert cfg.openai.api_key_env == "OPENAI_API_KEY"
    assert isinstance(cfg.openai.extra_headers, dict)


def test_openai_config_from_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "provider: openai\n"
        "openai:\n"
        "  model: llama-3.3-70b-versatile\n"
        "  base_url: https://api.groq.com/openai/v1\n"
        "  api_key_env: GROQ_API_KEY\n"
    )
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.provider == "openai"
    assert cfg.openai.model == "llama-3.3-70b-versatile"
    assert cfg.openai.base_url == "https://api.groq.com/openai/v1"
    assert cfg.openai.api_key_env == "GROQ_API_KEY"


def test_provider_defaults_to_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import load_config
    assert load_config().provider == "ollama"


def test_td_classifier_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from code_review_agent.config import load_config
    cfg = load_config()
    assert cfg.tools.td_classifier.model_path == "karths/binary_classification_train_TD"
    assert cfg.tools.td_classifier.backend == "auto"
    assert cfg.tools.td_classifier.device == "cpu"


# ---------------------------------------------------------------------------
# flatten_thresholds / get_thresholds_flat
# ---------------------------------------------------------------------------

def test_flatten_thresholds_nested():
    from code_review_agent.config import flatten_thresholds
    raw = {"A": {"value": 10, "explanation": "x"}, "B": {"value": 0.5}}
    assert flatten_thresholds(raw) == {"A": 10, "B": 0.5}


def test_flatten_thresholds_scalar_passthrough():
    from code_review_agent.config import flatten_thresholds
    raw = {"A": 10, "B": 0.5}
    assert flatten_thresholds(raw) == {"A": 10, "B": 0.5}


def test_flatten_thresholds_empty():
    from code_review_agent.config import flatten_thresholds
    assert flatten_thresholds({}) == {}
    assert flatten_thresholds(None) == {}


def test_get_thresholds_flat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "code_smells:\n  LONG_METHOD_LINES:\n    value: 40\n    explanation: foo\n"
    )
    from code_review_agent.config import load_config, get_thresholds_flat
    cfg = load_config()
    flat = get_thresholds_flat(cfg, "code_smells")
    assert flat == {"LONG_METHOD_LINES": 40}
    assert all(not isinstance(v, dict) for v in flat.values())


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------

def test_resolve_api_key_explicit(monkeypatch):
    from code_review_agent.config import resolve_api_key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key("sk-explicit", "OPENAI_API_KEY") == "sk-explicit"


def test_resolve_api_key_named_env(monkeypatch):
    from code_review_agent.config import resolve_api_key
    monkeypatch.setenv("GROQ_API_KEY", "gsk-123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key(None, "GROQ_API_KEY") == "gsk-123"


def test_resolve_api_key_fallback_env(monkeypatch):
    from code_review_agent.config import resolve_api_key
    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    assert resolve_api_key(None, "MISSING_KEY", fallback_env="OPENAI_API_KEY") == "sk-fallback"


def test_resolve_api_key_none(monkeypatch):
    from code_review_agent.config import resolve_api_key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NOPE", raising=False)
    assert resolve_api_key(None, "NOPE", fallback_env="OPENAI_API_KEY") is None
