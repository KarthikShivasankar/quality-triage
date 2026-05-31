"""
Configuration loader for code_review_agent.

Searches for config.yaml in:
  1. Explicit path argument
  2. ./config.yaml (CWD)
  3. ~/.config/code_review_agent/config.yaml
  4. Package-bundled defaults (src/code_review_agent/config.yaml)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Typed sub-configs
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_tokens: int = 8192
    max_iterations: int = 25
    timeout: int = 120


@dataclass
class OpenAIConfig:
    """Generic OpenAI-compatible backend.

    Works with the official OpenAI API and any service that exposes an
    OpenAI-compatible ``/chat/completions`` endpoint: Groq, Together,
    OpenRouter, Fireworks, DeepInfra, Mistral, a local llama.cpp ``server``,
    vLLM, LM Studio, text-generation-webui, etc.

    The API key is resolved (in priority order) from:
      1. ``api_key`` set directly here (discouraged for secrets)
      2. the environment variable named by ``api_key_env``
      3. the generic ``OPENAI_API_KEY`` environment variable
    """

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 8192
    max_iterations: int = 25
    timeout: int = 180
    # Extra headers (e.g. OpenRouter requires HTTP-Referer / X-Title)
    extra_headers: dict = field(default_factory=dict)


@dataclass
class AnthropicConfig:
    model: str = "claude-opus-4-6"
    api_key: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 8192
    max_iterations: int = 20


@dataclass
class GithubConfig:
    clone_dir: str = "/tmp/code_review_repos"
    depth: int = 1
    timeout: int = 120


@dataclass
class TDClassifierConfig:
    # The TD model family is now a set of *binary, per-category* models on the
    # HuggingFace Hub under the ``karths/`` namespace. The default is the
    # general technical-debt detector.
    model_path: str = "karths/binary_classification_train_TD"
    # backend: "auto" (ONNX if available, else torch) | "onnx" | "torch"
    backend: str = "auto"
    device: str = "cpu"
    onnx_path: str | None = None
    batch_size: int = 32


@dataclass
class ToolsConfig:
    ignore_dirs: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "venv", ".venv", "node_modules",
        "dist", "build", ".tox", ".eggs", "htmlcov",
    ])
    read_file_max_lines: int = 500
    td_classifier: TDClassifierConfig = field(default_factory=TDClassifierConfig)


@dataclass
class CodeIntelConfig:
    max_file_size_kb: int = 500
    include_private_symbols: bool = False
    metrics_enabled: bool = True
    top_complexity_n: int = 15


@dataclass
class ReportConfig:
    output_dir: str = "./reports"
    default_format: str = "markdown"
    include_code_snippets: bool = True
    max_snippet_lines: int = 10
    min_severity: str = "low"
    open_after_write: bool = False


@dataclass
class ReviewConfig:
    """User-controllable defaults for what a review runs and how it fixes.

    ``checks`` selects which detector families run; ``td_categories`` selects
    which technical-debt category models are used. CLI flags override these.
    Fixes are SUGGEST-ONLY by default — applying requires explicit opt-in.
    """

    # Detector families: ml | code | architectural | structural | td | code-intel
    checks: list[str] = field(default_factory=lambda: [
        "ml", "code", "architectural", "structural", "td", "code-intel",
    ])
    td_categories: list[str] = field(default_factory=lambda: ["general"])
    min_severity: str = "low"
    output_format: str = "markdown"   # markdown | json
    suggest_fixes: bool = False       # ask the agent for machine-applicable fixes
    apply_fixes: bool = False         # NEVER write without explicit confirmation


@dataclass
class AppConfig:
    provider: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    github: GithubConfig = field(default_factory=GithubConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    code_intel: CodeIntelConfig = field(default_factory=CodeIntelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    _raw: dict = field(default_factory=dict, repr=False)
    _source: str = field(default="defaults", repr=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def _find_config(explicit: str | None = None) -> tuple[dict, str]:
    """Return (parsed_yaml_dict, source_path). Falls back to {} if none found."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        Path.cwd() / "config.yaml",
        Path.home() / ".config" / "code_review_agent" / "config.yaml",
        Path(__file__).parent / "config.yaml",  # bundled defaults
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}, str(p)
    return {}, "defaults"


def _dc(cls, raw: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | None = None) -> AppConfig:
    raw, source = _find_config(path)

    def sub(key: str, cls, transform=None):
        d = raw.get(key, {})
        if transform:
            d = transform(d)
        return _dc(cls, d) if isinstance(d, dict) else cls()

    tools_raw = raw.get("tools", {})
    td_raw = tools_raw.get("td_classifier", {})
    tools_cfg = ToolsConfig(
        ignore_dirs=tools_raw.get("ignore_dirs", ToolsConfig().ignore_dirs),
        read_file_max_lines=tools_raw.get("read_file_max_lines", 500),
        td_classifier=_dc(TDClassifierConfig, td_raw) if td_raw else TDClassifierConfig(),
    )

    return AppConfig(
        provider=raw.get("provider", "ollama"),
        ollama=sub("ollama", OllamaConfig),
        openai=sub("openai", OpenAIConfig),
        anthropic=sub("anthropic", AnthropicConfig),
        github=sub("github", GithubConfig),
        tools=tools_cfg,
        code_intel=sub("code_intel", CodeIntelConfig),
        report=sub("report", ReportConfig),
        review=sub("review", ReviewConfig),
        _raw=raw,
        _source=source,
    )


def get_thresholds(config: AppConfig, smell_type: str) -> dict[str, Any]:
    """
    Return the raw thresholds dict for a detector type.
    smell_type: "code_smells" | "architectural_smells" | "structural_smells"
    Returns {THRESHOLD_NAME: {"value": N, "explanation": "..."}, ...}
    """
    return config._raw.get(smell_type, {})


def flatten_thresholds(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the ``{NAME: {"value": N, "explanation": "..."}}`` config shape to
    the ``{NAME: N}`` shape the code_quality_analyzer detectors actually expect.

    Plain scalar values are passed through unchanged, so both shapes work.
    """
    flat: dict[str, Any] = {}
    for name, spec in (raw or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            flat[name] = spec["value"]
        else:
            flat[name] = spec
    return flat


def get_thresholds_flat(config: AppConfig, smell_type: str) -> dict[str, Any]:
    """Return thresholds in the flat ``{NAME: value}`` shape detectors require."""
    return flatten_thresholds(get_thresholds(config, smell_type))


def resolve_api_key(
    explicit: str | None,
    api_key_env: str | None,
    fallback_env: str = "OPENAI_API_KEY",
) -> str | None:
    """Resolve an API key from an explicit value, a named env var, or a fallback env var."""
    if explicit:
        return explicit
    if api_key_env:
        val = os.environ.get(api_key_env)
        if val:
            return val
    return os.environ.get(fallback_env)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: AppConfig | None = None
_config_path: str | None = None


def get_config(path: str | None = None) -> AppConfig:
    global _config, _config_path
    if _config is None or (path and path != _config_path):
        _config = load_config(path)
        _config_path = path
    return _config


def reset_config() -> None:
    """Force reload on next get_config() call. Useful in tests."""
    global _config, _config_path
    _config = None
    _config_path = None
