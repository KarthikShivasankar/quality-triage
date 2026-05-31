"""
Shared review orchestration used by BOTH the CLI and the FastAPI web UI.

Centralises: resolving the analysis selection, building a (selection-restricted)
agent, composing the review prompt context, and parsing fix suggestions from
the agent output. Keeping this here avoids duplicating logic across surfaces.
"""

from __future__ import annotations

from typing import Any

from code_review_agent import selection
from code_review_agent.fixes import FixSuggestion, parse_fix_blocks
from code_review_agent.tools import TOOL_DEFINITIONS_OPENAI


def resolve_selection(cfg, checks, td_categories) -> dict[str, Any]:
    """Resolve families + TD categories from CLI/web input, falling back to config."""
    raw_checks = list(checks) if checks else list(getattr(cfg.review, "checks", []) or [])
    families = selection.resolve_families(raw_checks)

    raw_cats = list(td_categories) if td_categories else list(
        getattr(cfg.review, "td_categories", []) or []
    )
    cats, unknown = selection.resolve_td_categories(raw_cats)
    return {
        "families": families,
        "skipped": selection.skipped_families(families),
        "td_categories": cats,
        "unknown_td_categories": unknown,
    }


def review_context(user_context: str, sel: dict, suggest_fixes: bool) -> str:
    """Compose the extra-context string injected into the review prompt."""
    note = selection.selection_note(
        sel["families"], sel["td_categories"], include_fix_instructions=suggest_fixes
    )
    parts = [p for p in [(user_context or "").strip(), note] if p]
    return "\n\n".join(parts)


def build_agent(
    cfg,
    provider: str | None,
    model: str | None,
    families: list[str] | None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Build a selection-restricted agent. Raises ValueError on missing keys.

    Used by the web UI directly; the CLI wraps this with friendly messaging.
    """
    from code_review_agent.agent import CodeReviewAgent

    provider = provider or cfg.provider
    tools = None
    allowed = None
    if families is not None:
        tools = selection.filter_openai_tools(TOOL_DEFINITIONS_OPENAI, families)
        allowed = selection.allowed_tool_names(families)

    return CodeReviewAgent(
        provider=provider,
        model=model if provider in ("anthropic", "openai") else None,
        ollama_model=model if provider == "ollama" else None,
        base_url=base_url,
        api_key=api_key,
        tools=tools,
        allowed_tool_names=allowed,
    )


def run_review(
    cfg,
    target: str,
    provider: str | None = None,
    model: str | None = None,
    checks: list[str] | None = None,
    td_categories: list[str] | None = None,
    suggest_fixes: bool = False,
    user_context: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a full (non-streaming) review. Returns text + parsed fixes + selection.

    This is the entry point the web UI calls; the CLI streams instead but shares
    the same selection/context/fix-parsing helpers.
    """
    sel = resolve_selection(cfg, checks, td_categories)
    agent = build_agent(cfg, provider, model, sel["families"], base_url=base_url, api_key=api_key)
    ctx = review_context(user_context, sel, suggest_fixes)
    text = agent.review_text(target, extra_context=ctx)
    fixes: list[FixSuggestion] = parse_fix_blocks(text) if suggest_fixes else []
    return {
        "text": text,
        "fixes": [f.to_dict() for f in fixes],
        "fix_objects": fixes,
        "selection": sel,
        "provider": provider or cfg.provider,
        "model": model,
        "target": target,
    }


def known_td_categories() -> list[str]:
    """Primary TD category keys for UI dropdowns."""
    from code_review_agent.tools import TD_PRIMARY_CATEGORIES
    return list(TD_PRIMARY_CATEGORIES)
