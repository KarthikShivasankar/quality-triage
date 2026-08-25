"""
Configuration loader for code_review_agent.

Searches for config.yaml in:
  1. Explicit path argument
  2. ./config.yaml (CWD)
  3. ~/.config/code_review_agent/config.yaml
  4. Package-bundled defaults (src/code_review_agent/config.yaml)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Typed sub-configs
# ---------------------------------------------------------------------------

# Small Ollama test model: Hugging Face GGUF via `ollama pull hf.co/...`
# ~1.7 GB, completion-only (no native tool calling). Hybrid review uses this.
DEFAULT_OLLAMA_NATIVE_MODEL = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
DEFAULT_LITELLM_MODEL = f"ollama/{DEFAULT_OLLAMA_NATIVE_MODEL}"

LITELLM_PROVIDER_PREFIXES = (
    "ollama_chat/",
    "ollama/",
    "openai/",
    "anthropic/",
    "groq/",
    "gemini/",
    "azure/",
)


def has_litellm_prefix(model: str) -> bool:
    """True when `model` already starts with a known LiteLLM provider prefix."""
    return model.startswith(LITELLM_PROVIDER_PREFIXES)


def as_litellm_ollama(model: str) -> str:
    """Ensure an Ollama tag is a LiteLLM ``ollama/<native>`` string.

    Hugging Face GGUF tags like ``hf.co/org/repo:Q4_K_M`` contain slashes
    but are *not* LiteLLM prefixes.
    """
    if has_litellm_prefix(model):
        return model
    return f"ollama/{model}"


def ollama_native_name(model: str) -> str:
    """Strip ``ollama/`` or ``ollama_chat/`` for native Ollama APIs."""
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


@dataclass
class LLMConfig:
    """LiteLLM client settings. `model` is a LiteLLM model string (provider/name)."""

    model: str = DEFAULT_LITELLM_MODEL
    api_base: str | None = "http://localhost:11434"
    api_key: str | None = None
    timeout: int = 120
    max_tokens: int = 8192
    max_iterations: int = 20
    temperature: float = 0.0
    num_retries: int = 2
    drop_params: bool = True
    fallbacks: list[str] = field(default_factory=list)


@dataclass
class GithubConfig:
    clone_dir: str = "/tmp/code_review_repos"
    depth: int = 1
    timeout: int = 120
    fetch_issues: bool = True
    issue_limit: int = 10


@dataclass
class TDClassifierConfig:
    model_path: str = "karths/binary_classification_train_TD"
    device: str = "cpu"
    batch_size: int = 32
    backend: str = "onnx"  # "onnx" (default, no PyTorch needed) | "torch"


@dataclass
class ToolsConfig:
    ignore_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            "node_modules",
            "dist",
            "build",
            ".tox",
            ".eggs",
            "htmlcov",
            ".mypy_cache",
            ".ruff_cache",
        ]
    )
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
    default_format: str = "markdown"  # markdown | json | sarif | both
    include_code_snippets: bool = True
    max_snippet_lines: int = 10
    min_severity: str = "low"  # critical | high | medium | low | info
    fail_on: str = "none"  # none | critical | high | medium | low | info
    open_after_write: bool = False


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    aliases: dict[str, str] = field(
        default_factory=lambda: {
            "local": DEFAULT_LITELLM_MODEL,
            "test": DEFAULT_LITELLM_MODEL,
            "cheap": "groq/llama-3.3-70b-versatile",
            "frontier": "anthropic/claude-sonnet-4-6",
        }
    )
    github: GithubConfig = field(default_factory=GithubConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    code_intel: CodeIntelConfig = field(default_factory=CodeIntelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    _raw: dict = field(default_factory=dict, repr=False)
    _source: str = field(default="defaults", repr=False)

    @property
    def provider(self) -> str:
        """Best-effort provider prefix from the LiteLLM model string."""
        model = self.llm.model or ""
        if "/" in model:
            return model.split("/", 1)[0]
        return "openai"


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
        Path(__file__).parent / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}, str(p)
    return {}, "defaults"


def _dc(cls, raw: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def _migrate_legacy_llm(raw: dict) -> dict:
    """Map deprecated provider/ollama/anthropic blocks onto `llm` if needed."""
    if "llm" in raw and isinstance(raw["llm"], dict) and raw["llm"].get("model"):
        return raw.get("llm") or {}

    llm: dict[str, Any] = dict(raw.get("llm") or {})
    provider = str(raw.get("provider") or "ollama").lower()
    ollama = raw.get("ollama") or {}
    anthropic = raw.get("anthropic") or {}

    if provider == "anthropic":
        model = anthropic.get("model", "claude-sonnet-4-6")
        llm.setdefault(
            "model",
            str(model) if has_litellm_prefix(str(model)) else f"anthropic/{model}",
        )
        for src_key, dst_key in (
            ("max_tokens", "max_tokens"),
            ("max_iterations", "max_iterations"),
        ):
            if src_key in anthropic:
                llm.setdefault(dst_key, anthropic[src_key])
    else:
        model = ollama.get("model", DEFAULT_OLLAMA_NATIVE_MODEL)
        llm.setdefault("model", as_litellm_ollama(str(model)))
        if "base_url" in ollama:
            base = str(ollama["base_url"]).rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            llm.setdefault("api_base", base)
        if "api_key" in ollama:
            llm.setdefault("api_key", ollama["api_key"])
        for src_key in ("max_tokens", "max_iterations", "timeout"):
            if src_key in ollama:
                llm.setdefault(src_key, ollama[src_key])
    return llm


def load_config(path: str | None = None) -> AppConfig:
    raw, source = _find_config(path)

    def sub(key: str, cls):
        d = raw.get(key, {})
        return _dc(cls, d) if isinstance(d, dict) else cls()

    tools_raw = raw.get("tools", {}) if isinstance(raw.get("tools"), dict) else {}
    td_raw = (
        tools_raw.get("td_classifier", {})
        if isinstance(tools_raw.get("td_classifier"), dict)
        else {}
    )
    tools_cfg = ToolsConfig(
        ignore_dirs=tools_raw.get("ignore_dirs", ToolsConfig().ignore_dirs),
        read_file_max_lines=tools_raw.get("read_file_max_lines", 500),
        td_classifier=_dc(TDClassifierConfig, td_raw)
        if td_raw
        else TDClassifierConfig(),
    )

    aliases_raw = raw.get("aliases")
    aliases = dict(AppConfig().aliases)
    if isinstance(aliases_raw, dict):
        aliases.update({str(k): str(v) for k, v in aliases_raw.items()})

    llm_raw = _migrate_legacy_llm(raw)

    return AppConfig(
        llm=_dc(LLMConfig, llm_raw) if llm_raw else LLMConfig(),
        aliases=aliases,
        github=sub("github", GithubConfig),
        tools=tools_cfg,
        code_intel=sub("code_intel", CodeIntelConfig),
        report=sub("report", ReportConfig),
        _raw=raw,
        _source=source,
    )


def get_thresholds(config: AppConfig, smell_type: str) -> dict[str, Any]:
    """
    Return detector thresholds for a smell type.

    YAML stores `{NAME: {value: N, explanation: "..."}}`. Detectors expect
    `{NAME: N}`, so nested `value` keys are flattened.
    """
    raw = config._raw.get(smell_type, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "value" in val:
            out[str(key)] = val["value"]
        else:
            out[str(key)] = val
    return out


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
