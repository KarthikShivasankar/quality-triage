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
    assert cfg.ollama.model == "qwen2.5-coder:7b"
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
